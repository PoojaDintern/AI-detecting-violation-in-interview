import os
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from extensions import db
from models import (InterviewSession, TestSubmission, Violation, GazeEvent,
                    DeviceAlert, JobPosting, JobApplication, User)

recruiter_bp = Blueprint('recruiter', __name__)


@recruiter_bp.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    rid = current_user.id
    recruiter_session_ids = [s.id for s in InterviewSession.query.filter_by(recruiter_id=rid).all()]
    recruiter_posting_ids = [p.id for p in JobPosting.query.filter_by(recruiter_id=rid).all()]

    candidate_count = len(set(
        a.candidate_id for a in JobApplication.query.filter(
            JobApplication.posting_id.in_(recruiter_posting_ids)
        ).all()
    )) if recruiter_posting_ids else 0

    total_submissions = TestSubmission.query.filter(
        TestSubmission.session_id.in_(recruiter_session_ids)
    ).count() if recruiter_session_ids else 0

    total_violations = Violation.query.filter(
        Violation.session_id.in_(recruiter_session_ids)
    ).count() if recruiter_session_ids else 0

    active_sessions = InterviewSession.query.filter(
        InterviewSession.recruiter_id == rid,
        InterviewSession.status == 'active'
    ).count()

    total_gaze = GazeEvent.query.filter(
        GazeEvent.session_id.in_(recruiter_session_ids)
    ).count() if recruiter_session_ids else 0

    total_device = DeviceAlert.query.filter(
        DeviceAlert.session_id.in_(recruiter_session_ids)
    ).count() if recruiter_session_ids else 0

    recent_submissions = TestSubmission.query.filter(
        TestSubmission.session_id.in_(recruiter_session_ids)
    ).order_by(TestSubmission.submitted_at.desc()).limit(20).all() if recruiter_session_ids else []

    recent_violations = Violation.query.filter(
        Violation.session_id.in_(recruiter_session_ids)
    ).order_by(Violation.timestamp.desc()).limit(30).all() if recruiter_session_ids else []

    active_sessions_list = InterviewSession.query.filter(
        InterviewSession.recruiter_id == rid,
        InterviewSession.status == 'active'
    ).all()

    return jsonify({'success': True,
        'stats': {
            'total_candidates': candidate_count,
            'total_submissions': total_submissions,
            'total_violations': total_violations,
            'active_sessions': active_sessions,
            'total_gaze_events': total_gaze,
            'total_device_alerts': total_device,
        },
        'recent_submissions': [s.to_dict() for s in recent_submissions],
        'recent_violations': [v.to_dict() for v in recent_violations],
        'active_sessions_list': [s.to_dict() for s in active_sessions_list],
    })


@recruiter_bp.route('/api/candidates')
@login_required
def get_candidates():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    rid = current_user.id
    session_candidate_ids = set(
        s.candidate_id for s in InterviewSession.query.filter_by(recruiter_id=rid).all()
    )
    recruiter_posting_ids = [p.id for p in JobPosting.query.filter_by(recruiter_id=rid).all()]
    posting_candidate_ids = set(
        a.candidate_id for a in JobApplication.query.filter(
            JobApplication.posting_id.in_(recruiter_posting_ids)
        ).all()
    ) if recruiter_posting_ids else set()

    all_candidate_ids = session_candidate_ids | posting_candidate_ids
    if not all_candidate_ids:
        return jsonify({'success': True, 'candidates': []})

    candidates = User.query.filter(
        User.id.in_(all_candidate_ids),
        User.role == 'candidate'
    ).all()

    recruiter_session_ids = set(
        s.id for s in InterviewSession.query.filter_by(recruiter_id=rid).all()
    )

    return jsonify({'success': True, 'candidates': [
        {'id': c.id, 'username': c.username, 'full_name': c.full_name,
         'email': c.email,
         'submissions': TestSubmission.query.filter(
             TestSubmission.user_id == c.id,
             TestSubmission.session_id.in_(recruiter_session_ids)
         ).count() if recruiter_session_ids else 0
        } for c in candidates
    ]})


@recruiter_bp.route('/api/recruiter/profile', methods=['GET'])
@login_required
def get_recruiter_profile():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    parts = (current_user.full_name or '').split(' ', 1)
    company_name = current_user.company_name or (current_user.full_name or current_user.username)
    return jsonify({
        'success': True,
        'username': current_user.username,
        'email': current_user.email,
        'full_name': current_user.full_name or '',
        'first_name': parts[0] if parts else '',
        'last_name': parts[1] if len(parts) > 1 else '',
        'company_name': company_name,
        'phone': current_user.phone or '',
        'logo_url': current_user.logo_url or '',
        'industry': current_user.company_industry or '',
        'company_size': current_user.company_size or '',
        'website': current_user.company_website or '',
        'about': current_user.company_about or '',
        'smtp_email': current_user.smtp_email or '',
        'has_smtp': bool(current_user.smtp_email and current_user.smtp_app_password),
        'system_email_configured': bool(os.environ.get('EMAIL_USER') and os.environ.get('EMAIL_PASS')),
    })


@recruiter_bp.route('/api/recruiter/profile', methods=['POST'])
@login_required
def save_recruiter_profile():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data         = request.get_json() or {}
    first_name   = data.get('first_name', '').strip()
    last_name    = data.get('last_name', '').strip()
    company_name = data.get('company_name', '').strip()
    cur_pass     = data.get('current_password', '').strip()
    new_pass     = data.get('new_password', '').strip()
    full_name    = f'{first_name} {last_name}'.strip()
    if full_name:
        current_user.full_name = full_name
    if new_pass:
        if not cur_pass:
            return jsonify({'success': False, 'message': 'Current password required to change password.'}), 400
        if not current_user.check_password(cur_pass):
            return jsonify({'success': False, 'message': 'Current password is incorrect.'}), 400
        if len(new_pass) < 8:
            return jsonify({'success': False, 'message': 'New password must be at least 8 characters.'}), 400
        current_user.set_password(new_pass)
    phone        = data.get('phone', '').strip()
    logo_url     = data.get('logo_url', '').strip()
    industry     = data.get('industry', '').strip()
    company_size = data.get('company_size', '').strip()
    website      = data.get('website', '').strip()
    about        = data.get('about', '').strip()

    if company_name:
        current_user.company_name = company_name
        JobPosting.query.filter_by(recruiter_id=current_user.id, is_active=True).update({'company_name': company_name})
    if phone:
        current_user.phone = phone
    if logo_url:
        current_user.logo_url = logo_url
    if industry:
        current_user.company_industry = industry
    if company_size:
        current_user.company_size = company_size
    if website:
        current_user.company_website = website
    if about:
        current_user.company_about = about
    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile saved successfully!'})


@recruiter_bp.route('/api/recruiter/email-settings', methods=['GET'])
@login_required
def get_email_settings():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    return jsonify({
        'success': True,
        'smtp_email': current_user.smtp_email or '',
        'has_password': bool(current_user.smtp_app_password),
        'system_email_configured': bool(os.environ.get('EMAIL_USER') and os.environ.get('EMAIL_PASS')),
    })


@recruiter_bp.route('/api/recruiter/email-settings', methods=['POST'])
@login_required
def save_email_settings():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data       = request.get_json() or {}
    smtp_email = data.get('smtp_email', '').strip()
    smtp_pass  = data.get('smtp_app_password', '').strip()

    if smtp_email:
        import re
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', smtp_email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400

    current_user.smtp_email = smtp_email or None
    if smtp_pass:
        current_user.smtp_app_password = smtp_pass
    elif not smtp_email:
        current_user.smtp_app_password = None
    db.session.commit()

    if smtp_email and smtp_pass:
        import smtplib
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(smtp_email, smtp_pass)
            return jsonify({'success': True, 'message': '✅ Email settings saved and verified! Your Gmail is connected.'})
        except Exception as e:
            return jsonify({'success': True, 'verified': False,
                'message': f'⚠️ Settings saved but Gmail test failed: {str(e)}. Check your App Password.'})

    return jsonify({'success': True, 'message': 'Email settings saved.'})


# ── System Config API ─────────────────────────────────────────────────────────
@recruiter_bp.route('/api/system-config', methods=['GET'])
@login_required
def get_system_config():
    """Return current system configuration status."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    return jsonify({
        'success': True,
        'gemini_configured': bool(os.environ.get('GEMINI_API_KEY')),
        'email_configured':  bool(os.environ.get('EMAIL_USER') and os.environ.get('EMAIL_PASS')),
        'google_oauth':      bool(os.environ.get('GOOGLE_CLIENT_ID')),
        'github_oauth':      bool(os.environ.get('GITHUB_CLIENT_ID')),
        'plugin_key':        bool(os.environ.get('PLUGIN_API_KEY')),
        'base_url_set':      bool(os.environ.get('BASE_URL', '').startswith('http')),
        'base_url':          os.environ.get('BASE_URL', 'Not set'),
        'otp_email':         current_user.smtp_email or os.environ.get('EMAIL_USER', ''),
    })


@recruiter_bp.route('/api/system-config', methods=['POST'])
@login_required
def save_system_config():
    """Save OTP email configuration — stored in recruiter's account."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    import smtplib
    data      = request.get_json() or {}
    otp_email = data.get('otp_email', '').strip()
    otp_pass  = data.get('otp_pass', '').strip()

    if not otp_email and not otp_pass:
        current_user.smtp_email        = None
        current_user.smtp_app_password = None
        db.session.commit()
        return jsonify({'success': True, 'message': 'Cleared'})

    if not otp_email or not otp_pass:
        return jsonify({'success': False, 'message': 'Both email and password required'}), 400

    # Test connection
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
            srv.login(otp_email, otp_pass)
    except smtplib.SMTPAuthenticationError:
        return jsonify({'success': False,
                        'message': 'Authentication failed — check your App Password'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

    current_user.smtp_email        = otp_email
    current_user.smtp_app_password = otp_pass
    db.session.commit()
    return jsonify({'success': True, 'message': 'OTP email configured successfully'})


# ── Gemini Token Usage — DB backed ───────────────────────────────────────────
from models.token_usage import TokenUsage
from sqlalchemy import func


@recruiter_bp.route('/api/token-usage', methods=['GET'])
@login_required
def get_token_usage():
    """Return token usage totals and recent log from DB."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        # Aggregated totals by call_type
        rows = db.session.query(
            TokenUsage.call_type,
            func.sum(TokenUsage.total_tokens).label('total'),
            func.sum(TokenUsage.prompt_tokens).label('prompt'),
            func.sum(TokenUsage.response_tokens).label('response'),
            func.count(TokenUsage.id).label('calls')
        ).group_by(TokenUsage.call_type).all()

        totals = {}
        for r in rows:
            totals[r.call_type] = {
                'total': r.total or 0,
                'prompt': r.prompt or 0,
                'response': r.response or 0,
                'calls': r.calls or 0
            }

        grand_total = sum(v['total'] for v in totals.values())

        # Recent 50 log entries
        recent = TokenUsage.query.order_by(
            TokenUsage.called_at.desc()
        ).limit(50).all()

        return jsonify({
            'success':     True,
            'questions':   totals.get('questions',    {}).get('total', 0) + totals.get('mcq_questions', {}).get('total', 0),
            'evaluations': totals.get('evaluations',  {}).get('total', 0),
            'final':       totals.get('final',        {}).get('total', 0),
            'total':       grand_total,
            'breakdown':   totals,
            'log': [
                {
                    'type':   r.call_type,
                    'label':  r.label or '',
                    'tokens': r.total_tokens,
                    'prompt': r.prompt_tokens,
                    'response': r.response_tokens,
                    'time':   r.called_at.strftime('%d %b %H:%M') if r.called_at else '',
                }
                for r in recent
            ]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@recruiter_bp.route('/api/token-usage/reset', methods=['POST'])
@login_required
def reset_token_usage():
    """Delete all token usage records from DB."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        TokenUsage.query.delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@recruiter_bp.route('/api/token-usage/detail', methods=['GET'])
@login_required
def get_token_usage_detail():
    """Return full per-call breakdown with filters."""
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    call_type = request.args.get('type')
    limit     = int(request.args.get('limit', 100))
    q = TokenUsage.query.order_by(TokenUsage.called_at.desc())
    if call_type:
        q = q.filter_by(call_type=call_type)
    records = q.limit(limit).all()
    return jsonify({
        'success': True,
        'records': [r.to_dict() for r in records],
        'count':   len(records)
    })