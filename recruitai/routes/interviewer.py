import os
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from extensions import db
from models import InterviewerAssignment, InterviewSession, ScheduledInterview, User
from utils.email import send_interviewer_assignment_email

interviewer_bp = Blueprint('interviewer', __name__)


@interviewer_bp.route('/api/assign-interviewer', methods=['POST'])
@login_required
def assign_interviewer():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    session_id        = data.get('session_id')
    scheduled_id      = data.get('scheduled_id')
    interviewer_name  = (data.get('interviewer_name') or '').strip()
    interviewer_email = (data.get('interviewer_email') or '').strip().lower()
    round_number      = data.get('round_number', 1)
    round_name        = data.get('round_name', f'Round {round_number}')
    notes             = data.get('notes', '')

    if not session_id or not interviewer_name or not interviewer_email:
        return jsonify({'success': False, 'message': 'session_id, interviewer_name and interviewer_email are required'}), 400

    sess = db.session.get(InterviewSession, session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Interview session not found'}), 404

    interviewer_user = User.query.filter_by(email=interviewer_email).first()
    assignment = InterviewerAssignment(
        session_id=session_id, scheduled_id=scheduled_id,
        recruiter_id=current_user.id,
        interviewer_name=interviewer_name, interviewer_email=interviewer_email,
        interviewer_user_id=interviewer_user.id if interviewer_user else None,
        round_number=round_number, round_name=round_name, notes=notes, status='assigned',
    )
    db.session.add(assignment)
    db.session.commit()

    base_url   = os.environ.get('BASE_URL', 'http://localhost:5000')
    portal_url = f"{base_url}/interviewer_portal.html?assignment={assignment.id}&email={interviewer_email}"
    room_url   = f"{base_url}/recruiter_room.html?room={sess.room_code}&interviewer={assignment.id}&email={interviewer_email}"
    send_interviewer_assignment_email(
        to_email=interviewer_email, to_name=interviewer_name,
        recruiter_name=current_user.full_name or current_user.username,
        candidate_name=sess.candidate.full_name or sess.candidate.username,
        job_role=sess.job_role, round_name=round_name,
        room_code=sess.room_code, portal_url=portal_url, room_url=room_url, notes=notes,
        recruiter_smtp=current_user.smtp_email, recruiter_pass=current_user.smtp_app_password,
    )
    assignment.email_sent = True
    db.session.commit()

    return jsonify({'success': True, 'assignment': assignment.to_dict(),
                    'message': f'Interviewer {interviewer_name} assigned and notified via email.',
                    'portal_url': portal_url})


@interviewer_bp.route('/api/interviewer-assignments/<int:session_id>')
@login_required
def get_interviewer_assignments(session_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    assignments = InterviewerAssignment.query.filter_by(session_id=session_id).all()
    return jsonify({'success': True, 'assignments': [a.to_dict() for a in assignments]})


@interviewer_bp.route('/api/my-interviewer-assignments')
@login_required
def my_interviewer_assignments():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    assignments = InterviewerAssignment.query.filter_by(
        recruiter_id=current_user.id
    ).order_by(InterviewerAssignment.created_at.desc()).all()
    return jsonify({'success': True, 'assignments': [a.to_dict() for a in assignments]})


@interviewer_bp.route('/api/interviewer-room-access')
def interviewer_room_access():
    assignment_id = request.args.get('assignment', type=int)
    email         = (request.args.get('email') or '').strip().lower()
    if not assignment_id or not email:
        return jsonify({'success': False, 'message': 'assignment and email required'}), 400
    assignment = db.session.get(InterviewerAssignment, assignment_id)
    if not assignment:
        return jsonify({'success': False, 'message': 'Assignment not found'}), 404
    if assignment.interviewer_email.lower() != email:
        return jsonify({'success': False, 'message': 'Email does not match'}), 403
    sess = assignment.session
    if not sess:
        return jsonify({'success': False, 'message': 'Interview session not found'}), 404
    candidate = db.session.get(User, sess.candidate_id)
    if assignment.status == 'assigned':
        assignment.status = 'seen'
        db.session.commit()
    return jsonify({
        'success': True,
        'is_interviewer': True,
        'assignment_id': assignment_id,
        'interviewer_name': assignment.interviewer_name,
        'interviewer_email': assignment.interviewer_email,
        'session': {
            'id':              sess.id,
            'job_role':        sess.job_role,
            'mode':            sess.mode,
            'room_code':       sess.room_code,
            'status':          sess.status,
            'candidate_name':  (candidate.full_name or candidate.username) if candidate else 'Candidate',
            'recruiter_notes': assignment.notes or '',
            'round_number':    sess.round_number or 1,
            'round_name':      sess.round_name or 'Round 1',
        },
        'ice_servers': current_app.config.get('WEBRTC_ICE_SERVERS', []),
    })


@interviewer_bp.route('/api/interviewer-assignments-by-email')
def get_assignments_by_email():
    email = (request.args.get('email') or '').strip().lower()
    if not email:
        return jsonify({'success': False, 'message': 'email required'}), 400
    assignments = InterviewerAssignment.query.filter(
        db.func.lower(InterviewerAssignment.interviewer_email) == email
    ).order_by(InterviewerAssignment.created_at.desc()).all()
    if not assignments:
        return jsonify({'success': False, 'message': 'No assignments found for this email'}), 404
    base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
    result = []
    for a in assignments:
        sess = a.session
        if not sess:
            continue
        candidate = db.session.get(User, sess.candidate_id)
        sched = db.session.get(ScheduledInterview, a.scheduled_id) if a.scheduled_id else None
        portal_url = f"{base_url}/interviewer_portal.html?assignment={a.id}&email={email}"
        room_url   = f"{base_url}/recruiter_room.html?room={sess.room_code}&interviewer={a.id}&email={email}"
        result.append({
            'id':                a.id,
            'status':            a.status,
            'interviewer_name':  a.interviewer_name,
            'interviewer_email': a.interviewer_email,
            'round_number':      a.round_number,
            'round_name':        a.round_name or 'Round 1',
            'notes':             a.notes or '',
            'candidate_name':    (candidate.full_name or candidate.username) if candidate else 'Candidate',
            'job_role':          sess.job_role,
            'room_code':         sess.room_code,
            'portal_url':        portal_url,
            'room_url':          room_url,
            'scheduled_at':      sched.scheduled_at.strftime('%b %d, %Y • %I:%M %p') if sched else None,
        })
    interviewer_name = assignments[0].interviewer_name if assignments else ''
    return jsonify({'success': True, 'interviewer_name': interviewer_name, 'assignments': result})


@interviewer_bp.route('/api/interviewer-mark-seen/<int:assignment_id>', methods=['PATCH'])
def interviewer_mark_seen(assignment_id):
    data  = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    a = db.session.get(InterviewerAssignment, assignment_id)
    if not a:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    if a.interviewer_email.lower() != email:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    if a.status == 'assigned':
        a.status = 'seen'
        db.session.commit()
    return jsonify({'success': True})


@interviewer_bp.route('/api/interviewer-portal')
def interviewer_portal_data():
    assignment_id = request.args.get('assignment', type=int)
    email         = (request.args.get('email') or '').strip().lower()
    if not assignment_id or not email:
        return jsonify({'success': False, 'message': 'assignment and email parameters required'}), 400
    assignment = db.session.get(InterviewerAssignment, assignment_id)
    if not assignment:
        return jsonify({'success': False, 'message': 'Assignment not found'}), 404
    if assignment.interviewer_email.lower() != email:
        return jsonify({'success': False, 'message': 'Email does not match assignment'}), 403
    base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
    room_url = f"{base_url}/recruiter_room.html?room={assignment.session.room_code}&interviewer={assignment.id}&email={assignment.interviewer_email}"
    d = assignment.to_dict()
    d['room_url'] = room_url
    if assignment.scheduled_id:
        sched = db.session.get(ScheduledInterview, assignment.scheduled_id)
        if sched:
            d['scheduled_at'] = sched.scheduled_at.strftime('%B %d, %Y at %I:%M %p')
    if assignment.status == 'assigned':
        assignment.status = 'seen'
        db.session.commit()
    return jsonify({'success': True, 'assignment': d})


@interviewer_bp.route('/api/interviewer-submit-score', methods=['POST'])
def interviewer_submit_score():
    data          = request.get_json() or {}
    assignment_id = data.get('assignment_id')
    email         = (data.get('email') or '').strip().lower()
    score         = data.get('score')
    feedback      = data.get('feedback', '')
    if not assignment_id or not email or score is None:
        return jsonify({'success': False, 'message': 'assignment_id, email and score required'}), 400
    assignment = db.session.get(InterviewerAssignment, assignment_id)
    if not assignment:
        return jsonify({'success': False, 'message': 'Assignment not found'}), 404
    if assignment.interviewer_email.lower() != email:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        score = int(score)
        if not (0 <= score <= 100): raise ValueError
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Score must be 0–100'}), 400
    assignment.interviewer_score    = score
    assignment.interviewer_feedback = feedback
    assignment.status               = 'completed'
    sess = assignment.session
    if sess and not sess.interview_score:
        sess.interview_score = score
    if sess and feedback:
        note = f"[Interviewer: {assignment.interviewer_name}] {feedback}"
        sess.recruiter_notes = (sess.recruiter_notes + '\n' + note) if sess.recruiter_notes else note
    db.session.commit()
    return jsonify({'success': True, 'message': 'Score and feedback submitted successfully.'})


@interviewer_bp.route('/api/delete-interviewer-assignment/<int:assignment_id>', methods=['DELETE'])
@login_required
def delete_interviewer_assignment(assignment_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    assignment = db.session.get(InterviewerAssignment, assignment_id)
    if not assignment or assignment.recruiter_id != current_user.id:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    db.session.delete(assignment)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Assignment removed.'})
