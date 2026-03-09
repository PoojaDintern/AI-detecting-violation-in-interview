import re
from flask import Blueprint, request, jsonify, session, redirect
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from models import User, InterviewSession
from datetime import datetime

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    logout_user()
    session.clear()
    login_user(user, remember=False)
    redirect_url = 'test.html' if user.role == 'candidate' else 'recruiter_dashboard.html'
    return jsonify({'success': True, 'redirect': redirect_url,
                    'role': user.role, 'full_name': user.full_name})


@auth_bp.route('/signup', methods=['POST'])
def signup():
    data         = request.get_json() or {}
    username     = data.get('username', '').strip()
    email        = data.get('email', '').strip()
    password     = data.get('password', '')
    full_name    = data.get('full_name', '').strip()
    role         = data.get('role', 'candidate')
    company_name = data.get('company_name', '').strip()
    phone        = data.get('phone', '').strip()

    if not all([username, email, password, full_name]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400
    if len(full_name) < 2:
        return jsonify({'success': False, 'message': 'First name is required'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
    if len(username) < 3:
        return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return jsonify({'success': False, 'message': 'Username may only contain letters, numbers, and underscores'}), 400
    email_regex = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
    if not email_regex.match(email):
        return jsonify({'success': False, 'message': 'Please enter a valid email address'}), 400
    domain_part = email.split('@')[1]
    tld = domain_part.split('.')[-1].lower()
    invalid_tlds = {'invalid', 'test', 'localhost', 'fake', 'example', 'local'}
    if tld in invalid_tlds or len(tld) < 2:
        return jsonify({'success': False, 'message': 'Email domain appears invalid.'}), 400
    disposable_domains = {'mailinator.com', 'guerrillamail.com', 'temp-mail.org', 'throwaway.email',
                          'fakeinbox.com', 'sharklasers.com', 'trashmail.com', 'yopmail.com'}
    if domain_part.lower() in disposable_domains:
        return jsonify({'success': False, 'message': 'Disposable email addresses are not allowed.'}), 400
    if role in ('recruiter', 'admin'):
        if not company_name:
            return jsonify({'success': False, 'message': 'Company name is required for company accounts'}), 400
        if phone:
            phone_clean = re.sub(r'[\s\-().]', '', phone)
            if not re.match(r'^\+?[0-9]{7,15}$', phone_clean):
                return jsonify({'success': False, 'message': 'Please enter a valid phone number'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered. Please login.'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already taken'}), 400
    u = User(username=username, email=email, full_name=full_name, role=role)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Account created! Please login.'})


@auth_bp.route('/logout')
@login_required
def logout():
    active_sessions = InterviewSession.query.filter_by(
        candidate_id=current_user.id, status='active'
    ).all()
    for sess in active_sessions:
        sess.status = 'ended'
        sess.ended_at = datetime.utcnow()
    if active_sessions:
        db.session.commit()
    logout_user()
    return redirect('/')


@auth_bp.route('/api/check-auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'username': current_user.username,
                        'full_name': current_user.full_name, 'role': current_user.role,
                        'is_admin': current_user.is_admin,
                        'is_recruiter': current_user.is_recruiter})
    return jsonify({'authenticated': False})
