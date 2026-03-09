import os
import json
import random
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from extensions import db
from models import (JobPosting, JobApplication, ScheduledInterview, InterviewSession,
                    ExamQuestion, JobRoundConfig, RoundConfigDetail, InterviewerAssignment)
from utils.helpers import make_room_code, make_google_calendar_link
from utils.email import send_interview_email, send_interviewer_assignment_email

jobs_bp = Blueprint('jobs', __name__)

JOB_SECTIONS = [
    'IT', 'HR', 'Marketing', 'Finance', 'Sales', 'Operations',
    'Design', 'Product', 'Data & Analytics', 'Legal', 'Customer Support', 'Engineering'
]


@jobs_bp.route('/api/job-sections')
def get_job_sections():
    return jsonify({'success': True, 'sections': JOB_SECTIONS})


@jobs_bp.route('/api/job-postings', methods=['GET'])
@login_required
def list_job_postings():
    try:
        if current_user.is_admin or current_user.role == 'recruiter':
            postings = JobPosting.query.filter_by(recruiter_id=current_user.id, is_active=True).all()
        else:
            postings = JobPosting.query.filter_by(is_active=True).all()
        return jsonify({'success': True, 'postings': [p.to_dict() for p in postings]})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e), 'postings': []}), 500


@jobs_bp.route('/api/job-postings/all', methods=['GET'])
@login_required
def list_all_job_postings():
    if not (current_user.is_admin or current_user.role == 'recruiter'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        postings = JobPosting.query.filter_by(recruiter_id=current_user.id).order_by(JobPosting.created_at.desc()).all()
        return jsonify({'success': True, 'postings': [p.to_dict() for p in postings]})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e), 'postings': []}), 500


@jobs_bp.route('/api/job-postings', methods=['POST'])
@login_required
def create_job_posting():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    if not all([data.get('job_section'), data.get('job_role'), data.get('company_name')]):
        return jsonify({'success': False, 'message': 'job_section, job_role, and company_name are required'}), 400
    if not data.get('salary_package', '').strip():
        return jsonify({'success': False, 'message': 'Salary package is required'}), 400
    skills = data.get('skills_required', [])
    if not skills:
        return jsonify({'success': False, 'message': 'At least one skill is required'}), 400
    p = JobPosting(
        recruiter_id=current_user.id,
        job_section=data['job_section'],
        job_role=data['job_role'],
        job_title=data.get('job_title', ''),
        company_name=data['company_name'],
        description=data.get('description', ''),
        skills_required=json.dumps(skills),
        experience_required=data.get('experience_required', ''),
        job_type=data.get('job_type', ''),
        salary_package=data.get('salary_package', ''),
        work_mode=data.get('work_mode', ''),
    )
    db.session.add(p)
    db.session.commit()
    return jsonify({'success': True, 'posting': p.to_dict()}), 201


@jobs_bp.route('/api/job-postings/<int:posting_id>', methods=['DELETE'])
@login_required
def delete_job_posting(posting_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    p = db.session.get(JobPosting, posting_id)
    if not p or p.recruiter_id != current_user.id:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    p.is_active = False
    db.session.commit()
    return jsonify({'success': True})


@jobs_bp.route('/api/job-postings/<int:posting_id>/applications', methods=['GET'])
@login_required
def get_applications_for_posting(posting_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    p = db.session.get(JobPosting, posting_id)
    if not p or p.recruiter_id != current_user.id:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    apps = JobApplication.query.filter_by(posting_id=posting_id).order_by(JobApplication.applied_at.desc()).all()
    return jsonify({'success': True, 'applications': [a.to_dict() for a in apps]})


@jobs_bp.route('/api/applicants-by-role', methods=['GET'])
@login_required
def get_applicants_by_role():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    job_role = request.args.get('job_role', '').strip()
    if not job_role:
        return jsonify({'success': False, 'message': 'job_role is required'}), 400
    try:
        postings = JobPosting.query.filter_by(recruiter_id=current_user.id, job_role=job_role, is_active=True).all()
        if not postings:
            return jsonify({'success': True, 'candidates': [], 'posting_ids': []})
        posting_ids = [p.id for p in postings]
        apps = JobApplication.query.filter(
            JobApplication.posting_id.in_(posting_ids)
        ).order_by(JobApplication.applied_at.desc()).all()
        seen = set()
        candidates = []
        for a in apps:
            if a.candidate_id not in seen:
                seen.add(a.candidate_id)
                candidates.append({
                    'application_id': a.id,
                    'candidate_id': a.candidate_id,
                    'candidate_name': a.candidate.full_name or a.candidate.username,
                    'candidate_username': a.candidate.username,
                    'candidate_email': a.candidate.email,
                    'status': a.status,
                    'applied_at': a.applied_at.strftime('%Y-%m-%d'),
                    'posting_id': a.posting_id,
                    'company_name': a.posting.company_name,
                })
        return jsonify({'success': True, 'candidates': candidates})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e), 'candidates': []}), 500


@jobs_bp.route('/api/apply', methods=['POST'])
@login_required
def apply_for_job():
    if current_user.is_admin:
        return jsonify({'success': False, 'message': 'Recruiters cannot apply'}), 400
    data = request.get_json() or {}
    posting_id = data.get('posting_id')
    if not posting_id:
        return jsonify({'success': False, 'message': 'posting_id required'}), 400
    existing = JobApplication.query.filter_by(posting_id=posting_id, candidate_id=current_user.id).first()
    if existing:
        return jsonify({'success': False, 'message': 'Already applied for this position'}), 400
    app_obj = JobApplication(
        posting_id=posting_id,
        candidate_id=current_user.id,
        cover_note=data.get('cover_note', ''),
        status='applied',
    )
    db.session.add(app_obj)
    db.session.commit()
    return jsonify({'success': True, 'application': app_obj.to_dict()})


@jobs_bp.route('/api/my-applications')
@login_required
def my_applications():
    apps = JobApplication.query.filter_by(candidate_id=current_user.id).order_by(JobApplication.applied_at.desc()).all()
    result = []
    for a in apps:
        d = a.to_dict()
        sched = ScheduledInterview.query.filter_by(application_id=a.id).first()
        d['scheduled_interview'] = sched.to_dict() if sched else None
        result.append(d)
    return jsonify({'success': True, 'applications': result})


@jobs_bp.route('/api/my-scheduled-interviews')
@login_required
def my_scheduled_interviews():
    if current_user.is_admin:
        scheds = ScheduledInterview.query.filter_by(recruiter_id=current_user.id).order_by(ScheduledInterview.scheduled_at.desc()).all()
    else:
        scheds = ScheduledInterview.query.filter_by(candidate_id=current_user.id).order_by(ScheduledInterview.scheduled_at.desc()).all()
    return jsonify({'success': True, 'scheduled': [s.to_dict() for s in scheds]})


@jobs_bp.route('/api/schedule-interview', methods=['POST'])
@login_required
def schedule_interview():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    application_id    = data.get('application_id')
    scheduled_at_str  = data.get('scheduled_at')
    interview_mode    = data.get('interview_mode', 'mcq')
    round_number      = data.get('round_number', 1)
    round_name        = data.get('round_name', f'Round {round_number}')
    interviewer_name  = (data.get('interviewer_name') or '').strip()
    interviewer_email = (data.get('interviewer_email') or '').strip().lower()

    if not application_id or not scheduled_at_str:
        return jsonify({'success': False, 'message': 'application_id and scheduled_at required'}), 400

    app_obj = db.session.get(JobApplication, application_id)
    if not app_obj:
        return jsonify({'success': False, 'message': 'Application not found'}), 404

    try:
        scheduled_dt = datetime.fromisoformat(scheduled_at_str)
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid datetime format'}), 400

    posting   = app_obj.posting
    candidate = app_obj.candidate
    recruiter = current_user

    config = JobRoundConfig.query.filter_by(posting_id=posting.id).first()
    if config:
        detail = RoundConfigDetail.query.filter_by(config_id=config.id, round_number=round_number).first()
        if detail:
            interview_mode = detail.interview_mode
            round_name     = detail.round_name

    question_ids = []
    if interview_mode == 'mcq':
        qs = ExamQuestion.query.filter_by(job_role=posting.job_role).all()
        if len(qs) < 10:
            qs = ExamQuestion.query.limit(15).all()
        selected = random.sample(qs, min(10, len(qs)))
        question_ids = [q.id for q in selected]

    room_code = make_room_code()
    sess = InterviewSession(
        candidate_id=candidate.id, recruiter_id=recruiter.id,
        job_role=posting.job_role, mode=interview_mode,
        room_code=room_code, status='pending', credibility_score=100,
        question_ids=json.dumps(question_ids),
        round_number=round_number, round_name=round_name, posting_id=posting.id,
    )
    db.session.add(sess)
    db.session.commit()

    # Ensure pipeline
    from routes.sessions import _ensure_pipeline
    _ensure_pipeline(candidate.id, posting.id, round_number, round_name, interview_mode, sess.id)

    base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
    title    = f"RecruitAI Interview — {candidate.full_name or candidate.username} with {recruiter.full_name or recruiter.username}"
    desc     = f"Job Role: {posting.job_role} at {posting.company_name}\nRound: {round_name}\nRoom Code: {room_code}\nJoin: {base_url}/interview_room.html?room={room_code}"
    cal_link = make_google_calendar_link(title, desc, scheduled_dt)

    sched = ScheduledInterview(
        application_id=application_id, session_id=sess.id,
        recruiter_id=recruiter.id, candidate_id=candidate.id,
        scheduled_at=scheduled_dt, interview_mode=interview_mode,
        calendar_link=cal_link, room_code=room_code,
        round_number=round_number, round_name=round_name,
    )
    db.session.add(sched)
    app_obj.status = 'interview_scheduled'
    db.session.commit()

    scheduled_at_display = scheduled_dt.strftime('%B %d, %Y at %I:%M %p')
    r_smtp_user = recruiter.smtp_email or None
    r_smtp_pass = recruiter.smtp_app_password or None

    send_interview_email(
        candidate.email, candidate.full_name or candidate.username,
        recruiter.full_name or recruiter.username,
        posting.company_name, posting.job_role,
        scheduled_at_display, room_code, cal_link,
        role='candidate', smtp_user=r_smtp_user, smtp_pass=r_smtp_pass
    )
    send_interview_email(
        recruiter.email, candidate.full_name or candidate.username,
        recruiter.full_name or recruiter.username,
        posting.company_name, posting.job_role,
        scheduled_at_display, room_code, cal_link,
        role='recruiter', smtp_user=r_smtp_user, smtp_pass=r_smtp_pass
    )
    sched.email_sent = True
    db.session.commit()

    portal_url = None
    if interviewer_name and interviewer_email:
        from models import User
        interviewer_user = User.query.filter_by(email=interviewer_email).first()
        assignment = InterviewerAssignment(
            session_id=sess.id, scheduled_id=sched.id,
            recruiter_id=recruiter.id,
            interviewer_name=interviewer_name, interviewer_email=interviewer_email,
            interviewer_user_id=interviewer_user.id if interviewer_user else None,
            round_number=round_number, round_name=round_name, status='assigned',
        )
        db.session.add(assignment)
        db.session.commit()
        portal_url = f"{base_url}/interviewer_portal.html?assignment={assignment.id}&email={interviewer_email}"
        room_url   = f"{base_url}/recruiter_room.html?room={room_code}&interviewer={assignment.id}&email={interviewer_email}"
        send_interviewer_assignment_email(
            to_email=interviewer_email, to_name=interviewer_name,
            recruiter_name=recruiter.full_name or recruiter.username,
            candidate_name=candidate.full_name or candidate.username,
            job_role=posting.job_role, round_name=round_name,
            room_code=room_code, portal_url=portal_url, room_url=room_url,
            recruiter_smtp=r_smtp_user, recruiter_pass=r_smtp_pass,
        )
        assignment.email_sent = True
        db.session.commit()

    return jsonify({
        'success': True,
        'scheduled': sched.to_dict(),
        'room_code': room_code,
        'calendar_link': cal_link,
        'portal_url': portal_url,
        'message': f'Interview scheduled for {scheduled_at_display}. Emails sent to all parties.',
    })
