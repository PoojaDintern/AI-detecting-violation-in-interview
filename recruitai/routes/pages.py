from flask import Blueprint, send_from_directory, redirect
from flask_login import login_required, current_user
import os

pages_bp = Blueprint('pages', __name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')


@pages_bp.route('/')
def serve_index():
    return send_from_directory(TEMPLATES_DIR, 'index.html')


@pages_bp.route('/favicon.ico')
def favicon():
    return '', 204


@pages_bp.route('/index.html')
def serve_index_html():
    return send_from_directory(TEMPLATES_DIR, 'index.html')


# Candidate-only pages
@pages_bp.route('/test.html')
@login_required
def serve_test():
    if current_user.role in ('admin', 'recruiter'):
        return redirect('/recruiter_dashboard.html')
    return send_from_directory(TEMPLATES_DIR, 'test.html')


@pages_bp.route('/dashboard.html')
@login_required
def serve_dashboard():
    if current_user.role in ('admin', 'recruiter'):
        return redirect('/recruiter_dashboard.html')
    return send_from_directory(TEMPLATES_DIR, 'dashboard.html')


@pages_bp.route('/results.html')
@login_required
def serve_results():
    if current_user.role in ('admin', 'recruiter'):
        return redirect('/recruiter_dashboard.html')
    return send_from_directory(TEMPLATES_DIR, 'results.html')


@pages_bp.route('/interview_room.html')
@login_required
def serve_interview_room():
    if current_user.role in ('admin', 'recruiter'):
        return redirect('/recruiter_dashboard.html')
    return send_from_directory(TEMPLATES_DIR, 'interview_room.html')


@pages_bp.route('/interview_complete.html')
@login_required
def serve_interview_complete():
    if current_user.role in ('admin', 'recruiter'):
        return redirect('/recruiter_dashboard.html')
    return send_from_directory(TEMPLATES_DIR, 'interview_complete.html')


# Recruiter-only pages
@pages_bp.route('/recruiter_dashboard.html')
@login_required
def serve_recruiter_dashboard():
    if current_user.role not in ('admin', 'recruiter'):
        return redirect('/test.html')
    return send_from_directory(TEMPLATES_DIR, 'recruiter_dashboard.html')


@pages_bp.route('/recruiter_room.html')
@login_required
def serve_recruiter_room():
    if current_user.role not in ('admin', 'recruiter'):
        return redirect('/test.html')
    return send_from_directory(TEMPLATES_DIR, 'recruiter_room.html')


# Public pages (no login required)
@pages_bp.route('/interviewer_portal.html')
def serve_interviewer_portal():
    return send_from_directory(TEMPLATES_DIR, 'interviewer_portal.html')


@pages_bp.route('/interviewer_login.html')
def serve_interviewer_login():
    return send_from_directory(TEMPLATES_DIR, 'interviewer_login.html')
