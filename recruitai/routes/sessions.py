import json
import random
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from extensions import db, socketio
from models import (InterviewSession, ExamQuestion, User, InterviewPipeline,
                    JobRoundConfig, RoundConfigDetail, JobApplication,
                    CandidateCooldown, TestSubmission, Violation)
from utils.helpers import make_room_code
from utils.ai import gemini_ask

sessions_bp = Blueprint('sessions', __name__)

# Per-session consecutive miss counters
_face_miss_counter = {}
_gaze_away_counter = {}


def _ensure_pipeline(candidate_id, posting_id, round_number, round_name, mode, session_id):
    """Create or update the InterviewPipeline record for this candidate+posting+round."""
    try:
        config = JobRoundConfig.query.filter_by(posting_id=posting_id).first()
        total_rounds = config.total_rounds if config else 3
        config_id    = config.id if config else None

        app_obj = JobApplication.query.filter_by(
            candidate_id=candidate_id, posting_id=posting_id
        ).first()
        application_id = app_obj.id if app_obj else None

        pipeline = InterviewPipeline.query.filter_by(
            candidate_id=candidate_id, posting_id=posting_id
        ).first()

        if not pipeline:
            rounds = {}
            for rn in range(1, total_rounds + 1):
                detail = None
                if config:
                    detail = RoundConfigDetail.query.filter_by(config_id=config.id, round_number=rn).first()
                rounds[str(rn)] = {
                    'status': 'pending' if rn == 1 else 'locked',
                    'round_name': detail.round_name if detail else f'Round {rn}',
                    'mode': detail.interview_mode if detail else 'mcq',
                    'session_id': session_id if rn == round_number else None,
                    'submission_id': None,
                    'score': None,
                }
            pipeline = InterviewPipeline(
                candidate_id=candidate_id, posting_id=posting_id,
                application_id=application_id, config_id=config_id,
                total_rounds=total_rounds, current_round=round_number,
                overall_status='in_progress',
            )
            pipeline.set_rounds(rounds)
            db.session.add(pipeline)
        else:
            rounds = pipeline.get_rounds()
            rkey = str(round_number)
            if rkey not in rounds:
                rounds[rkey] = {}
            rounds[rkey]['session_id'] = session_id
            rounds[rkey]['status'] = 'pending'
            rounds[rkey]['round_name'] = round_name
            rounds[rkey]['mode'] = mode
            pipeline.set_rounds(rounds)
            pipeline.current_round = round_number
            pipeline.updated_at = datetime.utcnow()
        db.session.commit()
        return pipeline
    except Exception as e:
        db.session.rollback()
        print(f'Pipeline init error: {e}')
        return None


def _apply_cooldown(candidate_id, posting_id):
    from datetime import timedelta
    try:
        existing = CandidateCooldown.query.filter_by(candidate_id=candidate_id).first()
        eligible = datetime.utcnow() + timedelta(days=60)
        if existing:
            existing.triggered_at = datetime.utcnow()
            existing.eligible_at  = eligible
            existing.is_active    = True
            existing.triggered_by_posting_id = posting_id
        else:
            cooldown = CandidateCooldown(
                candidate_id=candidate_id,
                triggered_by_posting_id=posting_id,
                triggered_at=datetime.utcnow(),
                eligible_at=eligible,
                is_active=True,
            )
            db.session.add(cooldown)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f'Cooldown apply error: {e}')


def calc_credibility(session_id):
    DEVICE_VIOLATION_TYPES = {'phone_detected', 'device_detected'}
    violations = Violation.query.filter_by(session_id=session_id).all()
    scored = [v for v in violations if v.violation_type not in DEVICE_VIOLATION_TYPES]
    score = 100
    severity_points = current_app.config.get('SEVERITY_POINTS', {1: 5, 2: 10, 3: 15})
    for v in scored:
        score -= severity_points.get(v.severity, 5)
    if len(scored) > 10:
        score -= (len(scored) - 10) * 2
    return max(0, min(100, score))


@sessions_bp.route('/api/job-roles')
def get_job_roles():
    db_roles = db.session.query(ExamQuestion.job_role).distinct().all()
    roles = [r[0] for r in db_roles]
    for r in current_app.config.get('JOB_ROLES', []):
        if r not in roles:
            roles.append(r)
    return jsonify({'success': True, 'roles': sorted(roles)})


@sessions_bp.route('/api/mcq-questions/<job_role>')
@login_required
def get_mcq_questions(job_role):
    questions = ExamQuestion.query.filter_by(job_role=job_role).all()
    if len(questions) < 10:
        questions = ExamQuestion.query.limit(10).all()
    selected = random.sample(questions, min(10, len(questions)))
    return jsonify({'success': True, 'questions': [q.to_dict() for q in selected], 'job_role': job_role})


@sessions_bp.route('/api/create-session', methods=['POST'])
@login_required
def create_session():
    data         = request.get_json() or {}
    job_role     = data.get('job_role', 'Python Developer')
    mode         = data.get('mode', 'mcq')
    posting_id   = data.get('posting_id')
    round_number = data.get('round_number', 1)
    round_name   = data.get('round_name', f'Round {round_number}')
    candidate_username = data.get('candidate_username')

    if current_user.role == 'candidate':
        candidate    = current_user
        recruiter_id = None
    else:
        if candidate_username:
            candidate = User.query.filter_by(username=candidate_username).first()
            if not candidate:
                return jsonify({'success': False, 'message': 'Candidate not found'}), 404
        else:
            return jsonify({'success': False, 'message': 'candidate_username required'}), 400
        recruiter_id = current_user.id

    if posting_id and current_user.is_recruiter:
        config = JobRoundConfig.query.filter_by(posting_id=posting_id).first()
        if config:
            detail = RoundConfigDetail.query.filter_by(
                config_id=config.id, round_number=round_number
            ).first()
            if detail:
                mode       = detail.interview_mode
                round_name = detail.round_name

    skip_scan             = bool(data.get('skip_scan', False))
    skip_device_detection = bool(data.get('skip_device_detection', False))
    mcq_source = data.get('mcq_source', 'hardcoded')
    question_ids = []
    if mode == 'mcq' and mcq_source != 'ai':
        qs = ExamQuestion.query.filter_by(job_role=job_role).all()
        if len(qs) < 10:
            qs = ExamQuestion.query.limit(15).all()
        selected = random.sample(qs, min(10, len(qs)))
        question_ids = [q.id for q in selected]

    sess = InterviewSession(
        candidate_id=candidate.id, recruiter_id=recruiter_id,
        job_role=job_role, mode=mode,
        room_code=make_room_code(), status='pending',
        credibility_score=100, question_ids=json.dumps(question_ids),
        mcq_source=mcq_source if mode == 'mcq' else None,
        round_number=round_number, round_name=round_name,
        posting_id=posting_id,
        recruiter_notes=json.dumps({'skip_scan': skip_scan, 'skip_device': skip_device_detection}) if (skip_scan or skip_device_detection) else None,
    )
    db.session.add(sess)
    db.session.commit()

    if posting_id:
        _ensure_pipeline(candidate.id, posting_id, round_number, round_name, mode, sess.id)

    return jsonify({'success': True, 'session': sess.to_dict(), 'room_code': sess.room_code})


def _get_skip_scan(sess):
    """Check if 360 scan is disabled for this session."""
    try:
        if sess.recruiter_notes:
            notes = json.loads(sess.recruiter_notes)
            return bool(notes.get('skip_scan', False))
    except Exception:
        pass
    return False


def _get_skip_device(sess):
    """Check if device detection is disabled for this session."""
    try:
        if sess.recruiter_notes:
            notes = json.loads(sess.recruiter_notes)
            return bool(notes.get('skip_device', False))
    except Exception:
        pass
    return False


@sessions_bp.route('/api/join-session/<room_code>')
def join_session(room_code):
    try:
        if not current_user.is_authenticated:
            return jsonify({'success': False, 'code': 'not_authenticated',
                            'message': 'Please log in to join this session.'}), 401

        sess = InterviewSession.query.filter_by(room_code=room_code).first()
        if not sess:
            return jsonify({'success': False, 'message': 'Session not found'}), 404

        if current_user.is_recruiter:
            return jsonify({'success': True, 'session': sess.to_dict(),
                            'skip_scan': _get_skip_scan(sess),
                            'skip_device_detection': _get_skip_device(sess),
                            'questions': [], 'ice_servers': current_app.config.get('WEBRTC_ICE_SERVERS', [])})

        if sess.candidate_id and sess.candidate_id != current_user.id:
            return jsonify({'success': False,
                            'message': 'This session was not assigned to your account. Make sure you are logged in with the correct candidate account that received the interview invitation.'}), 403

        if sess.status in ('completed', 'abandoned'):
            return jsonify({'success': False, 'message': 'already_submitted'}), 400

        try:
            if sess.posting_id and sess.round_number:
                pipeline = InterviewPipeline.query.filter_by(
                    candidate_id=current_user.id, posting_id=sess.posting_id
                ).first()
                if pipeline:
                    rounds = pipeline.get_rounds()
                    rkey = str(sess.round_number)
                    if rkey in rounds:
                        rdata = rounds[rkey]
                        if rdata.get('status') in ('completed_passed', 'completed_failed'):
                            return jsonify({
                                'success': False,
                                'message': 'round_already_completed',
                                'round_number': sess.round_number,
                                'round_name': sess.round_name or f'Round {sess.round_number}',
                                'score': rdata.get('score', 0),
                                'passed': rdata.get('status') == 'completed_passed',
                            }), 400
        except Exception:
            pass

        questions = []
        try:
            mcq_src = getattr(sess, 'mcq_source', 'hardcoded') or 'hardcoded'
            if sess.mode == 'mcq':
                if mcq_src == 'ai' and not sess.question_ids:
                    try:
                        prompt = (
                                f"Generate exactly 10 multiple-choice questions for a {sess.job_role} interview. "
                                "Return ONLY a JSON array of objects with keys: question_text, options (object with a,b,c,d), correct_answer (letter a/b/c/d). "
                                "No markdown, no explanation, just pure JSON array."
                            )
                        raw_text = gemini_ask(
                                current_app._get_current_object(), prompt,
                                track_type='mcq_questions',
                                track_label=sess.job_role,
                                session_id=sess.id
                            )
                        if raw_text:
                            raw = raw_text.strip().lstrip('```').lstrip('json').strip()
                            ai_qs = json.loads(raw)
                            questions = [
                                {
                                    'id': -(i + 1),
                                    'question_text': q.get('question_text', q.get('question', '')),
                                    'options': q.get('options', {'a': '', 'b': '', 'c': '', 'd': ''}),
                                    'correct_answer': q.get('correct_answer', 'a'),
                                    'difficulty': 'medium', 'category': sess.job_role,
                                }
                                for i, q in enumerate(ai_qs[:10])
                            ]
                        else:
                            raise Exception('Gemini unavailable')
                    except Exception as ae:
                        print(f'[join_session] AI MCQ failed: {ae}. Falling back to DB.')
                        qs = ExamQuestion.query.filter_by(job_role=sess.job_role).all()
                        if not qs:
                            qs = ExamQuestion.query.limit(15).all()
                        selected = random.sample(qs, min(10, len(qs)))
                        questions = [q.to_dict() for q in selected]
                        sess.question_ids = json.dumps([q.id for q in selected])
                        db.session.commit()
                elif sess.question_ids:
                    ids = json.loads(sess.question_ids)
                    questions = [db.session.get(ExamQuestion, i).to_dict() for i in ids
                                 if db.session.get(ExamQuestion, i)]
                else:
                    qs = ExamQuestion.query.filter_by(job_role=sess.job_role).all()
                    if not qs:
                        qs = ExamQuestion.query.limit(15).all()
                    selected = random.sample(qs, min(10, len(qs)))
                    questions = [q.to_dict() for q in selected]
                    sess.question_ids = json.dumps([q.id for q in selected])
                    db.session.commit()
        except Exception as qe:
            print(f'[join_session] Question load error: {qe}')
            questions = []

        return jsonify({'success': True, 'session': sess.to_dict(),
                        'skip_scan': _get_skip_scan(sess),
                        'skip_device_detection': _get_skip_device(sess),
                        'questions': questions,
                        'ice_servers': current_app.config.get('WEBRTC_ICE_SERVERS', [])})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error. Please try again.'}), 500


@sessions_bp.route('/api/candidate-session-access')
def candidate_session_access():
    room_code = request.args.get('room', '').strip().upper()
    email     = (request.args.get('email') or '').strip().lower()

    if not room_code:
        return jsonify({'success': False, 'message': 'room code required'}), 400

    sess = InterviewSession.query.filter_by(room_code=room_code).first()
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found.'}), 404

    if email:
        candidate = db.session.get(User, sess.candidate_id)
        if not candidate or candidate.email.lower() != email:
            return jsonify({'success': False, 'message': 'Email does not match this session.'}), 403

    if sess.status in ('completed', 'abandoned'):
        return jsonify({'success': False, 'message': 'already_submitted'}), 400

    return jsonify({
        'success': True,
        'room_code': sess.room_code,
        'job_role': sess.job_role,
        'mode': sess.mode,
        'session_id': sess.id,
        'candidate_email': db.session.get(User, sess.candidate_id).email if sess.candidate_id else '',
    })


@sessions_bp.route('/api/start-session/<int:session_id>', methods=['POST'])
@login_required
def start_session(session_id):
    sess = db.session.get(InterviewSession, session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    sess.status = 'active'
    sess.started_at = datetime.utcnow()
    db.session.commit()
    socketio.emit('session_started', {'session_id': session_id, 'room_code': sess.room_code}, room=sess.room_code)
    return jsonify({'success': True})


@sessions_bp.route('/api/session-end-on-close', methods=['POST'])
@login_required
def session_end_on_close():
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False}), 400
        sess = db.session.get(InterviewSession, session_id)
        if not sess:
            return jsonify({'success': False}), 404
        if sess.status == 'active':
            sess.status = 'abandoned'
            sess.ended_at = datetime.utcnow()
            db.session.commit()
            socketio.emit('session_abandoned', {
                'session_id': session_id,
                'candidate': sess.candidate.username,
                'room_code': sess.room_code,
            }, room='dashboard')
            socketio.emit('candidate_left', {
                'session_id': session_id, 'reason': 'tab_closed',
            }, room=sess.room_code)
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@sessions_bp.route('/api/retake-exam', methods=['POST'])
@login_required
def retake_exam():
    data     = request.get_json() or {}
    job_role = data.get('job_role', 'Python Developer')
    mode     = data.get('mode', 'mcq')
    qs = ExamQuestion.query.filter_by(job_role=job_role).all()
    if len(qs) < 10:
        qs = ExamQuestion.query.limit(15).all()
    selected = random.sample(qs, min(10, len(qs)))
    sess = InterviewSession(
        candidate_id=current_user.id, job_role=job_role, mode=mode,
        room_code=make_room_code(), status='pending', credibility_score=100,
        question_ids=json.dumps([q.id for q in selected]),
    )
    db.session.add(sess)
    db.session.commit()
    return jsonify({'success': True, 'session': sess.to_dict(),
                    'room_code': sess.room_code,
                    'message': 'Fresh session created. Credibility starts at 100.'})


@sessions_bp.route('/submit-test', methods=['POST'])
@login_required
def submit_test():
    data           = request.get_json() or {}
    session_id     = data.get('session_id')
    device_failed  = data.get('device_failed', False)
    cred_score     = calc_credibility(session_id) if session_id else 100

    db_sess      = db.session.get(InterviewSession, session_id) if session_id else None
    mode         = db_sess.mode         if db_sess else data.get('mode', 'mcq')
    job_role     = db_sess.job_role     if db_sess else data.get('job_role', 'General')
    posting_id   = db_sess.posting_id   if db_sess else None
    round_number = db_sess.round_number if db_sess else 1
    round_name   = db_sess.round_name   if db_sess else 'Round 1'

    interview_score = 0
    if mode == 'mcq':
        answers = data.get('answers', {})
        question_ids = []
        if session_id and db_sess and db_sess.question_ids:
            question_ids = json.loads(db_sess.question_ids)
        correct = 0
        for i, qid in enumerate(question_ids):
            q = db.session.get(ExamQuestion, qid)
            if q:
                user_ans = answers.get(f'q{i+1}', answers.get(str(qid), ''))
                if user_ans == q.correct_answer:
                    correct += 1
        interview_score = int((correct / max(len(question_ids), 1)) * 100)
    elif mode == 'ai_interview':
        interview_score = data.get('ai_overall_score', 0)

    pass_threshold = 50
    if posting_id:
        config = JobRoundConfig.query.filter_by(posting_id=posting_id).first()
        if config:
            detail = RoundConfigDetail.query.filter_by(
                config_id=config.id, round_number=round_number).first()
            if detail:
                pass_threshold = detail.pass_threshold

    passed = cred_score >= 50 and interview_score >= pass_threshold
    if device_failed:
        passed = False

    attempt_num = TestSubmission.query.filter_by(user_id=current_user.id).count() + 1

    from models import TestSubmission as TS
    sub = TS(
        user_id=current_user.id, session_id=session_id,
        job_role=job_role, mode=mode,
        answers=json.dumps(data.get('answers', {})),
        credibility_score=cred_score, interview_score=interview_score,
        total_violations=Violation.query.filter_by(session_id=session_id).count() if session_id else 0,
        exam_duration_seconds=data.get('duration_seconds', 0),
        passed=passed, attempt_number=attempt_num,
        ai_feedback=json.dumps(data.get('ai_feedback', {})),
        round_number=round_number, round_name=round_name, posting_id=posting_id,
    )
    db.session.add(sub)

    if session_id and db_sess:
        db_sess.status = 'completed'
        db_sess.ended_at = datetime.utcnow()
        db_sess.credibility_score = cred_score
        db_sess.interview_score   = interview_score
        # Link ScheduledInterview to this session so completed tab works
        from models import ScheduledInterview
        sched_link = ScheduledInterview.query.filter(
            ScheduledInterview.room_code == db_sess.room_code,
            ScheduledInterview.session_id == None
        ).first()
        if sched_link:
            sched_link.session_id = session_id
    db.session.commit()

    next_round_unlocked = False
    if posting_id:
        pipeline = InterviewPipeline.query.filter_by(
            candidate_id=current_user.id, posting_id=posting_id).first()
        if pipeline:
            rounds = pipeline.get_rounds()
            rkey = str(round_number)
            if rkey not in rounds:
                rounds[rkey] = {}
            rounds[rkey].update({
                'status': 'completed_passed' if passed else 'completed_failed',
                'submission_id': sub.id, 'score': interview_score,
                'credibility': cred_score, 'passed': passed,
            })
            total = pipeline.total_rounds
            if passed and round_number < total:
                nkey = str(round_number + 1)
                if rounds.get(nkey, {}).get('status') == 'locked':
                    rounds[nkey]['status'] = 'pending'
                    next_round_unlocked = True
                pipeline.current_round = round_number + 1
            all_done = all(
                rounds.get(str(rn), {}).get('status', 'locked')
                in ('completed_passed', 'completed_failed')
                for rn in range(1, total + 1)
            )
            if all_done:
                all_passed = all(
                    rounds.get(str(rn), {}).get('status') == 'completed_passed'
                    for rn in range(1, total + 1)
                )
                pipeline.overall_status = 'completed_passed' if all_passed else 'completed_failed'
                if not all_passed:
                    _apply_cooldown(current_user.id, posting_id)
            pipeline.set_rounds(rounds)
            pipeline.updated_at = datetime.utcnow()
            db.session.commit()

    return jsonify({'success': True, 'submission_id': sub.id,
                    'credibility_score': cred_score, 'interview_score': interview_score,
                    'total_violations': sub.total_violations, 'passed': passed,
                    'round_number': round_number, 'round_name': round_name,
                    'next_round_unlocked': next_round_unlocked,
                    'redirect': f'results.html?id={sub.id}'})


@sessions_bp.route('/api/room-status/<room_code>')
@login_required
def room_status(room_code):
    sess = InterviewSession.query.filter_by(room_code=room_code).first()
    if not sess:
        return jsonify({'success': False, 'message': 'Room not found'}), 404
    recruiter_present = sess.status == 'active'
    scheduled_at = sess.created_at.strftime('%Y-%m-%d %H:%M:%S') if sess.created_at else None
    return jsonify({
        'success': True,
        'recruiter_present': recruiter_present,
        'status': sess.status,
        'scheduled_at': scheduled_at,
        'job_role': sess.job_role,
        'mode': sess.mode,
    })


@sessions_bp.route('/api/interview-complete/<int:session_id>')
@login_required
def interview_complete_info(session_id):
    sess = db.session.get(InterviewSession, session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    if sess.candidate_id != current_user.id and not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    from models import ScheduledInterview, TestSubmission as TS
    sched = ScheduledInterview.query.filter_by(session_id=session_id).first()
    sub   = TS.query.filter_by(session_id=session_id).first()

    return jsonify({
        'success': True,
        'session': sess.to_dict(),
        'scheduled': sched.to_dict() if sched else None,
        'submission': sub.to_dict() if sub else None,
    })

@sessions_bp.route('/api/force-end-session', methods=['POST'])
@login_required
def force_end_session():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'message': 'session_id required'}), 400
    sess = db.session.get(InterviewSession, session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    if sess.recruiter_id != current_user.id and not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    sess.status = 'completed'
    sess.ended_at = datetime.utcnow()
    db.session.commit()
    socketio.emit('session_force_ended', {'session_id': session_id}, room=sess.room_code)
    return jsonify({'success': True})


@sessions_bp.route('/api/active-sessions', methods=['GET'])
@login_required
def get_active_sessions():
    """Get all currently active interview sessions for this recruiter."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    active = InterviewSession.query.filter_by(
        recruiter_id=current_user.id, status='active'
    ).all()
    result = []
    for s in active:
        from models.user import User
        candidate = db.session.get(User, s.candidate_id)
        result.append({
            'id': s.id,
            'room_code': s.room_code,
            'job_role': s.job_role,
            'mode': s.mode,
            'candidate_name': candidate.full_name or candidate.username if candidate else 'Unknown',
            'started_at': s.started_at.isoformat() if s.started_at else None,
        })
    return jsonify({'success': True, 'sessions': result, 'count': len(result)})


@sessions_bp.route('/api/extend-session', methods=['POST'])
@login_required
def extend_session():
    """Extend an active session by notifying candidate via socket."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data           = request.get_json() or {}
    session_id     = data.get('session_id')
    extend_minutes = int(data.get('extend_minutes', 15))
    if not session_id:
        return jsonify({'success': False, 'message': 'session_id required'}), 400
    sess = db.session.get(InterviewSession, session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    if sess.recruiter_id != current_user.id and not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    if sess.status != 'active':
        return jsonify({'success': False, 'message': 'Session is not active'}), 400
    # Notify candidate via socket
    socketio.emit('session_extended', {
        'session_id':     session_id,
        'extend_minutes': extend_minutes,
        'message':        f'Your interview time has been extended by {extend_minutes} minutes by the recruiter.'
    }, room=sess.room_code)
    return jsonify({'success': True, 'message': f'Session extended by {extend_minutes} minutes'})