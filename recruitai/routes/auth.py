import re
import os
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
    if role == 'recruiter':
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
    u = User(username=username, email=email, full_name=full_name, role=role, company_name=company_name if company_name else full_name)
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
                        'is_admin': current_user.is_recruiter,
                        'is_recruiter': current_user.is_recruiter,
                        'company_name': current_user.company_name or current_user.full_name or current_user.username,
                        'photo_url': current_user.photo_url or ''})
    return jsonify({'authenticated': False})


@auth_bp.route('/api/upload-photo', methods=['POST'])
@login_required
def upload_photo():
    data      = request.get_json() or {}
    photo_b64 = data.get('photo_url', '').strip()
    if not photo_b64:
        return jsonify({'success': False, 'message': 'No photo provided'}), 400
    if len(photo_b64) > 3_000_000:
        return jsonify({'success': False, 'message': 'Photo too large. Please use a smaller image.'}), 400
    try:
        current_user.photo_url = photo_b64
        db.session.commit()
        return jsonify({'success': True, 'message': 'Profile photo saved'})
    except Exception as e:
        db.session.rollback()
        err = str(e).lower()
        if 'column' in err and 'photo_url' in err:
            return jsonify({'success': False,
                'message': 'Database not ready — run this SQL in Supabase: ALTER TABLE users ADD COLUMN IF NOT EXISTS photo_url TEXT;'}), 500
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500


@auth_bp.route('/api/verify-face', methods=['POST'])
@login_required
def verify_face():
    """Compare a live camera frame against stored profile photo using pixel similarity."""
    import base64, struct
    data        = request.get_json() or {}
    live_frame  = data.get('live_frame', '')   # base64 image from camera
    stored_photo = current_user.photo_url or ''

    if not stored_photo:
        return jsonify({'success': False, 'match': False,
                        'message': 'No profile photo on file. Please upload your photo first.'}), 400
    if not live_frame:
        return jsonify({'success': False, 'match': False, 'message': 'No live frame provided'}), 400

    # Use OpenCV to compare faces if available
    try:
        import cv2
        import numpy as np

        def b64_to_img(b64str):
            if ',' in b64str:
                b64str = b64str.split(',')[1]
            img_bytes = base64.b64decode(b64str)
            arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        stored_img = b64_to_img(stored_photo)
        live_img   = b64_to_img(live_frame)

        if stored_img is None or live_img is None:
            return jsonify({'success': True, 'match': True, 'confidence': 0.5,
                            'message': 'Could not decode images — allowing entry'})

        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def get_face_crop(img):
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
            if len(faces) == 0:
                return None
            x,y,w,h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            return cv2.resize(img[y:y+h, x:x+w], (128,128))

        stored_face = get_face_crop(stored_img)
        live_face   = get_face_crop(live_img)

        if stored_face is None or live_face is None:
            return jsonify({'success': True, 'match': True, 'confidence': 0.5,
                            'message': 'Face not clearly detected — allowing entry'})

        # Compare using histogram similarity
        similarity_scores = []
        for ch in range(3):
            h1 = cv2.calcHist([stored_face],[ch],None,[64],[0,256])
            h2 = cv2.calcHist([live_face],[ch],None,[64],[0,256])
            cv2.normalize(h1,h1); cv2.normalize(h2,h2)
            similarity_scores.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

        confidence = float(np.mean(similarity_scores))
        match      = confidence >= 0.55   # threshold — 0.55 is reasonably strict

        return jsonify({
            'success':    True,
            'match':      match,
            'confidence': round(confidence, 3),
            'message':    'Identity verified' if match else 'Face does not match profile photo'
        })

    except ImportError:
        # OpenCV not available — allow entry with warning
        return jsonify({'success': True, 'match': True, 'confidence': 0.5,
                        'message': 'Face verification unavailable — allowing entry'})
    except Exception as e:
        print(f'Face verify error: {e}')
        return jsonify({'success': True, 'match': True, 'confidence': 0.5,
                        'message': 'Verification error — allowing entry'})


# ── OTP — Send verification code ───────────────────────────────────────────────
@auth_bp.route('/api/send-otp', methods=['POST'])
def send_otp():
    import random, time
    data     = request.get_json() or {}
    email    = data.get('email', '').strip().lower()
    username = data.get('username', '').strip()

    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400

    # Check if email or username already taken
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered. Please login.'}), 400
    if username and User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already taken'}), 400

    # Generate 6-digit OTP
    otp  = str(random.randint(100000, 999999))
    exp  = int(time.time()) + 600  # 10 minutes

    # Store in Flask session
    session['otp_code']  = otp
    session['otp_email'] = email
    session['otp_exp']   = exp

    # Send email
    email_configured = bool(os.environ.get('EMAIL_USER') and os.environ.get('EMAIL_PASS'))
    _send_otp_email(email, otp)

    print(f"[OTP] Sent {otp} to {email}")

    msg = 'Verification code sent to your email'
    if not email_configured:
        msg = 'Email not configured — check your terminal/console for the verification code'

    return jsonify({'success': True, 'message': msg, 'email_configured': email_configured})


def _send_otp_email(to_email, otp):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_user = db.session.execute(
        db.text("SELECT smtp_email FROM users WHERE role='recruiter' AND smtp_email IS NOT NULL LIMIT 1")
    ).fetchone()

    # Try system env first, then any recruiter's SMTP
    email_user = (os.environ.get('EMAIL_USER') or
                  (smtp_user[0] if smtp_user else None))
    email_pass = os.environ.get('EMAIL_PASS', '')

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px;background:#f5f7ff;border-radius:16px;">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:40px;">🤖</div>
        <h2 style="color:#0f172a;margin:8px 0 4px;">RecruitAI</h2>
        <p style="color:#64748b;font-size:14px;">Email Verification</p>
      </div>
      <div style="background:white;border-radius:12px;padding:28px;text-align:center;border:1px solid #dde3f4;">
        <p style="color:#1e293b;font-size:15px;margin-bottom:20px;">Your verification code is:</p>
        <div style="font-size:42px;font-weight:800;letter-spacing:12px;color:#4361ee;font-family:monospace;margin:16px 0;">{otp}</div>
        <p style="color:#64748b;font-size:13px;margin-top:16px;">This code expires in <strong>10 minutes</strong>.</p>
        <p style="color:#94a3b8;font-size:12px;margin-top:8px;">If you didn't request this, ignore this email.</p>
      </div>
    </div>
    """

    if email_user and email_pass:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = 'Your RecruitAI verification code'
            msg['From']    = email_user
            msg['To']      = to_email
            msg.attach(MIMEText(html, 'html'))
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
                srv.login(email_user, email_pass)
                srv.sendmail(email_user, [to_email], msg.as_string())
            print(f"[OTP] Email sent to {to_email}")
        except Exception as e:
            print(f"[OTP] Email failed: {e}")
    else:
        print(f"[OTP] No SMTP configured. Code for {to_email}: {otp}")


# ── OTP — Verify code ──────────────────────────────────────────────────────────
@auth_bp.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    import time
    data  = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    otp   = data.get('otp', '').strip()

    stored_otp   = session.get('otp_code')
    stored_email = session.get('otp_email', '').lower()
    stored_exp   = session.get('otp_exp', 0)

    if not stored_otp:
        return jsonify({'success': False, 'message': 'No verification code found. Please request a new one.'}), 400
    if int(time.time()) > stored_exp:
        session.pop('otp_code', None)
        return jsonify({'success': False, 'message': 'Code expired. Please request a new one.'}), 400
    if email != stored_email:
        return jsonify({'success': False, 'message': 'Email mismatch. Please request a new code.'}), 400
    if otp != stored_otp:
        return jsonify({'success': False, 'message': 'Incorrect code. Please check and try again.'}), 400

    # Clear OTP from session
    session.pop('otp_code', None)
    session.pop('otp_email', None)
    session.pop('otp_exp', None)

    return jsonify({'success': True, 'message': 'Email verified'})


# ── OAuth — Google & GitHub (simple redirect stubs) ───────────────────────────
@auth_bp.route('/auth/google')
def auth_google():
    """
    Google OAuth — requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env
    Install: pip install flask-dance
    For now returns a helpful message if not configured.
    """
    client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
    if not client_id:
        return """
        <html><body style="font-family:Arial;text-align:center;padding:60px;">
        <h2>Google OAuth not configured</h2>
        <p>Add <code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> to your <code>.env</code> file.</p>
        <p>See setup guide: <a href="https://console.cloud.google.com/apis/credentials">Google Cloud Console</a></p>
        <a href="/" style="color:#4361ee">← Back to login</a>
        </body></html>
        """
    # Redirect to Google OAuth
    from urllib.parse import urlencode
    base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
    params = {
        'client_id': client_id,
        'redirect_uri': f"{base_url}/auth/google/callback",
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
    }
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


@auth_bp.route('/auth/google/callback')
def auth_google_callback():
    import requests as req
    code         = request.args.get('code')
    client_id    = os.environ.get('GOOGLE_CLIENT_ID', '')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET', '')
    base_url     = os.environ.get('BASE_URL', 'http://localhost:5000')

    if not code or not client_id:
        return redirect('/?error=google_failed')

    try:
        # Exchange code for token
        token_res = req.post('https://oauth2.googleapis.com/token', data={
            'code': code, 'client_id': client_id, 'client_secret': client_secret,
            'redirect_uri': f"{base_url}/auth/google/callback", 'grant_type': 'authorization_code'
        }).json()

        access_token = token_res.get('access_token')
        if not access_token:
            return redirect('/?error=google_token_failed')

        # Get user info
        user_info = req.get('https://www.googleapis.com/oauth2/v2/userinfo',
                            headers={'Authorization': f'Bearer {access_token}'}).json()

        email     = user_info.get('email', '')
        full_name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('id', '')

        if not email:
            return redirect('/?error=google_no_email')

        # Find or create user
        user = User.query.filter_by(email=email).first()
        if not user:
            base_username = email.split('@')[0].replace('.', '_')
            username = base_username
            counter  = 1
            while User.query.filter_by(username=username).first():
                username = f"{base_username}{counter}"; counter += 1
            user = User(username=username, email=email, full_name=full_name,
                        role='candidate', password_hash='oauth_google_' + google_id)
            db.session.add(user)
            db.session.commit()

        login_user(user, remember=False)
        redirect_url = 'test.html' if user.role == 'candidate' else 'recruiter_dashboard.html'
        return redirect(f"/{redirect_url}")

    except Exception as e:
        print(f"Google OAuth error: {e}")
        return redirect('/?error=google_error')


@auth_bp.route('/auth/github')
def auth_github():
    client_id = os.environ.get('GITHUB_CLIENT_ID', '')
    if not client_id:
        return """
        <html><body style="font-family:Arial;text-align:center;padding:60px;">
        <h2>GitHub OAuth not configured</h2>
        <p>Add <code>GITHUB_CLIENT_ID</code> and <code>GITHUB_CLIENT_SECRET</code> to your <code>.env</code> file.</p>
        <p>See setup guide: <a href="https://github.com/settings/developers">GitHub Developer Settings</a></p>
        <a href="/" style="color:#4361ee">← Back to login</a>
        </body></html>
        """
    from urllib.parse import urlencode
    base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
    params   = {'client_id': client_id, 'redirect_uri': f"{base_url}/auth/github/callback", 'scope': 'user:email'}
    return redirect(f"https://github.com/login/oauth/authorize?{urlencode(params)}")


@auth_bp.route('/auth/github/callback')
def auth_github_callback():
    import requests as req
    code          = request.args.get('code')
    client_id     = os.environ.get('GITHUB_CLIENT_ID', '')
    client_secret = os.environ.get('GITHUB_CLIENT_SECRET', '')
    base_url      = os.environ.get('BASE_URL', 'http://localhost:5000')

    if not code or not client_id:
        return redirect('/?error=github_failed')

    try:
        token_res = req.post('https://github.com/login/oauth/access_token', data={
            'client_id': client_id, 'client_secret': client_secret,
            'code': code, 'redirect_uri': f"{base_url}/auth/github/callback"
        }, headers={'Accept': 'application/json'}).json()

        access_token = token_res.get('access_token')
        if not access_token:
            return redirect('/?error=github_token_failed')

        headers   = {'Authorization': f'token {access_token}'}
        user_info = req.get('https://api.github.com/user', headers=headers).json()
        emails    = req.get('https://api.github.com/user/emails', headers=headers).json()

        email = next((e['email'] for e in emails if e.get('primary') and e.get('verified')), None)
        if not email:
            email = user_info.get('email') or f"{user_info.get('login','user')}@github.local"

        full_name = user_info.get('name') or user_info.get('login', 'GitHub User')
        github_id = str(user_info.get('id', ''))

        user = User.query.filter_by(email=email).first()
        if not user:
            base_username = (user_info.get('login') or email.split('@')[0]).replace('-', '_')
            username = base_username; counter = 1
            while User.query.filter_by(username=username).first():
                username = f"{base_username}{counter}"; counter += 1
            user = User(username=username, email=email, full_name=full_name,
                        role='candidate', password_hash='oauth_github_' + github_id)
            db.session.add(user)
            db.session.commit()

        login_user(user, remember=False)
        redirect_url = 'test.html' if user.role == 'candidate' else 'recruiter_dashboard.html'
        return redirect(f"/{redirect_url}")

    except Exception as e:
        print(f"GitHub OAuth error: {e}")
        return redirect('/?error=github_error')