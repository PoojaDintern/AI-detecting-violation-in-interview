# app.py - Main Flask Application - NO TEMPLATES VERSION
from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from config import Config
import cv2
import base64
import numpy as np
import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

# ========================================
# INITIALIZE FLASK APP (ONLY ONCE!)
# ========================================
app = Flask(__name__, static_folder='.')
app.config.from_object(Config)
Config.init_app(app)

# Add CORS support
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# ========================================
# DATABASE MODELS
# ========================================

class User(UserMixin, db.Model):
    """User model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(512), nullable=False)
    full_name = db.Column(db.String(150))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    violations = db.relationship('Violation', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    submissions = db.relationship('TestSubmission', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password using scrypt"""
        self.password_hash = generate_password_hash(password, method='scrypt')
        print(f"✅ Password set for {self.username}")
    
    def check_password(self, password):
        """Verify password"""
        result = check_password_hash(self.password_hash, password)
        print(f"🔐 Password check for '{self.username}': {'✅ MATCH' if result else '❌ NO MATCH'}")
        return result
    
    def __repr__(self):
        return f'<User {self.username}>'

class Violation(db.Model):
    """Violation model"""
    __tablename__ = 'violations'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    violation_type = db.Column(db.String(50), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    severity = db.Column(db.Integer, default=1, nullable=False)
    description = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username,
            'violation_type': self.violation_type,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'severity': self.severity,
            'description': self.description
        }
    
    def __repr__(self):
        return f'<Violation {self.violation_type}>'

class TestSubmission(db.Model):
    """Test submission model"""
    __tablename__ = 'test_submissions'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    answers = db.Column(db.Text, nullable=False)
    credibility_score = db.Column(db.Integer, default=100, nullable=False)
    total_violations = db.Column(db.Integer, default=0, nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    exam_duration_seconds = db.Column(db.Integer)
    passed = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username,
            'full_name': self.user.full_name,
            'answers': json.loads(self.answers),
            'credibility_score': self.credibility_score,
            'total_violations': self.total_violations,
            'submitted_at': self.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
            'exam_duration_seconds': self.exam_duration_seconds,
            'passed': self.passed
        }

# ========================================
# FLASK-LOGIN SETUP
# ========================================

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    """Handle unauthorized access"""
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Login required'}), 401
    return send_from_directory('.', 'index.html')

# ========================================
# DATABASE INITIALIZATION
# ========================================

def init_database():
    """Initialize database"""
    try:
        with app.app_context():
            db.create_all()
            print("✅ Database tables created!")
            
            # Create admin user if doesn't exist
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@proctoring.com',
                    full_name='Administrator',
                    is_admin=True
                )
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                print("✅ Admin user created! (username: admin, password: admin123)")
            
            # Create test student if doesn't exist
            student = User.query.filter_by(username='student').first()
            if not student:
                student = User(
                    username='student',
                    email='student@test.com',
                    full_name='Test Student',
                    is_admin=False
                )
                student.set_password('student123')
                db.session.add(student)
                db.session.commit()
                print("✅ Test student created! (username: student, password: student123)")
            
            return True
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return False

# Initialize database
with app.app_context():
    init_database()

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ========================================
# HELPER FUNCTIONS
# ========================================

def log_violation_to_db(user_id, violation_type, severity, description):
    """Log violation to database"""
    try:
        violation = Violation(
            user_id=user_id,
            violation_type=violation_type,
            severity=severity,
            description=description
        )
        db.session.add(violation)
        db.session.commit()
        
        # Emit real-time alert to dashboard
        socketio.emit('violation_alert', {
            'user_id': user_id,
            'username': User.query.get(user_id).username,
            'violation_type': violation_type,
            'severity': severity,
            'timestamp': datetime.utcnow().strftime('%H:%M:%S')
        }, room='dashboard')
        
        return violation
    except Exception as e:
        db.session.rollback()
        print(f"Error logging violation: {str(e)}")
        return None

def get_violation_info(violation_type):
    """Get violation severity and description"""
    violation_map = {
        'tab_switch': {'severity': 2, 'description': 'User switched tab/window'},
        'exit_fullscreen': {'severity': 3, 'description': 'User exited fullscreen'},
        'no_face': {'severity': 2, 'description': 'Face not visible'},
        'multiple_faces': {'severity': 3, 'description': 'Multiple faces detected'},
        'right_click': {'severity': 1, 'description': 'Right-click attempt'},
        'copy_attempt': {'severity': 1, 'description': 'Copy attempt'},
        'paste_attempt': {'severity': 1, 'description': 'Paste attempt'},
        'devtools': {'severity': 2, 'description': 'DevTools attempt'}
    }
    return violation_map.get(violation_type, {'severity': 1, 'description': 'Unknown'})

def calculate_credibility_score(violations):
    """Calculate credibility score"""
    base_score = 100
    for v in violations:
        base_score -= app.config['SEVERITY_POINTS'].get(v.severity, 5)
    
    # Extra penalty for too many violations
    if len(list(violations)) > 10:
        base_score -= (len(list(violations)) - 10) * 2
    
    return max(0, min(100, base_score))

def generate_pdf_report(submission):
    """Generate PDF report for submission"""
    try:
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(1*inch, height - 1*inch, "Exam Proctoring Report")
        
        # User info
        pdf.setFont("Helvetica", 12)
        y = height - 1.5*inch
        pdf.drawString(1*inch, y, f"Student: {submission.user.full_name}")
        y -= 0.3*inch
        pdf.drawString(1*inch, y, f"Username: {submission.user.username}")
        y -= 0.3*inch
        pdf.drawString(1*inch, y, f"Email: {submission.user.email}")
        y -= 0.3*inch
        pdf.drawString(1*inch, y, f"Submitted: {submission.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Credibility Score
        y -= 0.5*inch
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(1*inch, y, f"Credibility Score: {submission.credibility_score}/100")
        
        # Status
        y -= 0.4*inch
        pdf.setFont("Helvetica", 14)
        status = "PASSED ✓" if submission.passed else "FAILED ✗"
        color = colors.green if submission.passed else colors.red
        pdf.setFillColor(color)
        pdf.drawString(1*inch, y, f"Status: {status}")
        pdf.setFillColor(colors.black)
        
        # Violations
        y -= 0.5*inch
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(1*inch, y, f"Total Violations: {submission.total_violations}")
        
        y -= 0.4*inch
        pdf.setFont("Helvetica", 10)
        violations = Violation.query.filter_by(user_id=submission.user_id).all()
        
        for v in violations[:15]:
            y -= 0.25*inch
            if y < 1*inch:
                pdf.showPage()
                y = height - 1*inch
            severity_text = {1: 'Low', 2: 'Medium', 3: 'High'}[v.severity]
            pdf.drawString(1*inch, y, f"• {v.timestamp.strftime('%H:%M:%S')} - {v.violation_type} (Severity: {severity_text})")
        
        pdf.setFont("Helvetica", 8)
        pdf.drawString(1*inch, 0.5*inch, "Generated by AI Proctoring System")
        
        pdf.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

# ========================================
# ROUTES - STATIC FILE SERVING
# ========================================

@app.route('/')
def index():
    """Home page - serve index.html"""
    if current_user.is_authenticated:
        if current_user.is_admin:
            return send_from_directory('.', 'dashboard.html')
        else:
            return send_from_directory('.', 'test.html')
    return send_from_directory('.', 'index.html')

@app.route('/index.html')
def serve_index():
    """Serve index.html directly"""
    return send_from_directory('.', 'index.html')

@app.route('/test.html')
@login_required
def serve_test():
    """Serve test.html"""
    if current_user.is_admin:
        return send_from_directory('.', 'dashboard.html')
    return send_from_directory('.', 'test.html')

@app.route('/dashboard.html')
@login_required
def serve_dashboard():
    """Serve dashboard.html"""
    if not current_user.is_admin:
        return send_from_directory('.', 'test.html')
    return send_from_directory('.', 'dashboard.html')

@app.route('/results.html')
@login_required
def serve_results():
    """Serve results.html"""
    return send_from_directory('.', 'results.html')

# ========================================
# ROUTES - AUTHENTICATION
# ========================================

@app.route('/login', methods=['POST'])
def login():
    """Login endpoint"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'message': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        print(f"\n{'='*60}")
        print(f"🔐 LOGIN ATTEMPT")
        print(f"{'='*60}")
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password)}")
        
        if not username or not password:
            print("❌ Missing credentials")
            return jsonify({
                'success': False,
                'message': 'Username and password are required'
            }), 400
        
        user = User.query.filter_by(username=username).first()
        
        if not user:
            print(f"❌ User '{username}' not found")
            return jsonify({
                'success': False,
                'message': 'Invalid username or password'
            }), 401
        
        print(f"✅ User found: {user.username}")
        
        if not user.check_password(password):
            print(f"❌ Password incorrect")
            return jsonify({
                'success': False,
                'message': 'Invalid username or password'
            }), 401
        
        login_user(user, remember=True)
        print(f"✅ Login successful for {user.username}")
        
        # Return HTML file names for redirect
        redirect_page = 'dashboard.html' if user.is_admin else 'test.html'
        print(f"🔀 Redirect to: {redirect_page}")
        
        response = jsonify({
            'success': True,
            'message': 'Login successful',
            'is_admin': user.is_admin,
            'redirect': redirect_page
        })
        response.status_code = 200
        return response
        
    except Exception as e:
        print(f"\n❌ LOGIN ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/signup', methods=['POST'])
def signup():
    """Signup endpoint"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'message': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        
        if not all([username, email, password, full_name]):
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400
        
        if len(password) < 6:
            return jsonify({
                'success': False,
                'message': 'Password must be at least 6 characters'
            }), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({
                'success': False,
                'message': 'Username already exists'
            }), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({
                'success': False,
                'message': 'Email already exists'
            }), 400
        
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            is_admin=False
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        print(f"✅ New user created: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully! Please login.'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Signup error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error creating account: {str(e)}'
        }), 500

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    logout_user()
    return redirect('/')

@app.route('/api/check-auth')
def check_auth():
    """Check authentication status"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'username': current_user.username,
            'is_admin': current_user.is_admin,
            'full_name': current_user.full_name
        })
    return jsonify({'authenticated': False})

# ========================================
# ROUTES - EXAM
# ========================================

@app.route('/exam')
@login_required
def exam():
    """Exam page redirect"""
    if current_user.is_admin:
        return send_from_directory('.', 'dashboard.html')
    return send_from_directory('.', 'test.html')

@app.route('/detect-face', methods=['POST'])
@login_required
def detect_face():
    """Face detection endpoint"""
    try:
        data = request.get_json()
        image_data = data['image']
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        num_faces = len(faces)
        
        if num_faces == 0:
            log_violation_to_db(current_user.id, 'no_face', 2, 'Face not visible')
        elif num_faces > 1:
            log_violation_to_db(current_user.id, 'multiple_faces', 3, f'{num_faces} faces')
        
        return jsonify({
            'success': True,
            'face_detected': num_faces == 1,
            'num_faces': num_faces,
            'message': f'{"Face detected" if num_faces == 1 else "No face" if num_faces == 0 else "Multiple faces"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/log-violation', methods=['POST'])
@login_required
def log_violation():
    """Log violation"""
    try:
        data = request.get_json()
        violation_type = data.get('violation_type')
        info = get_violation_info(violation_type)
        log_violation_to_db(current_user.id, violation_type, info['severity'], info['description'])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/submit-test', methods=['POST'])
@login_required
def submit_test():
    """Submit test"""
    try:
        data = request.get_json()
        violations = Violation.query.filter_by(user_id=current_user.id).all()
        credibility_score = calculate_credibility_score(violations)
        passed = credibility_score >= app.config['CREDIBILITY_PASS_THRESHOLD']
        
        submission = TestSubmission(
            user_id=current_user.id,
            answers=json.dumps(data.get('answers', {})),
            credibility_score=credibility_score,
            total_violations=len(violations),
            exam_duration_seconds=data.get('duration_seconds', 0),
            passed=passed
        )
        
        db.session.add(submission)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'submission_id': submission.id,
            'credibility_score': credibility_score,
            'total_violations': len(violations),
            'passed': passed,
            'redirect': f'results.html?id={submission.id}'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-credibility-score')
@login_required
def get_credibility_score():
    """Get current credibility score"""
    violations = Violation.query.filter_by(user_id=current_user.id).all()
    score = calculate_credibility_score(violations)
    return jsonify({
        'success': True,
        'credibility_score': score,
        'total_violations': len(violations)
    })

# ========================================
# ROUTES - RESULTS
# ========================================

@app.route('/api/results/<int:submission_id>')
@login_required
def get_results(submission_id):
    """Get results data as JSON"""
    submission = TestSubmission.query.get_or_404(submission_id)
    
    if submission.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    violations = Violation.query.filter_by(user_id=submission.user_id).all()
    
    breakdown = {}
    for v in violations:
        breakdown[v.violation_type] = breakdown.get(v.violation_type, 0) + 1
    
    return jsonify({
        'success': True,
        'submission': submission.to_dict(),
        'violations': [v.to_dict() for v in violations],
        'breakdown': breakdown
    })

@app.route('/download-report/<int:submission_id>')
@login_required
def download_report(submission_id):
    """Download PDF report"""
    submission = TestSubmission.query.get_or_404(submission_id)
    
    if submission.user_id != current_user.id and not current_user.is_admin:
        return "Unauthorized", 403
    
    pdf_buffer = generate_pdf_report(submission)
    
    if pdf_buffer:
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'exam_report_{submission.user.username}_{submission.id}.pdf',
            mimetype='application/pdf'
        )
    else:
        return "Error generating PDF", 500

# ========================================
# ROUTES - ADMIN DASHBOARD
# ========================================

@app.route('/dashboard')
@login_required
def dashboard():
    """Admin dashboard redirect"""
    if not current_user.is_admin:
        return send_from_directory('.', 'test.html')
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    """Get dashboard statistics"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    total_students = User.query.filter_by(is_admin=False).count()
    total_submissions = TestSubmission.query.count()
    total_violations = Violation.query.count()
    active_exams = 0
    
    recent_submissions = TestSubmission.query.order_by(
        TestSubmission.submitted_at.desc()
    ).limit(10).all()
    
    recent_violations = Violation.query.order_by(
        Violation.timestamp.desc()
    ).limit(20).all()
    
    return jsonify({
        'success': True,
        'stats': {
            'total_students': total_students,
            'total_submissions': total_submissions,
            'total_violations': total_violations,
            'active_exams': active_exams
        },
        'recent_submissions': [s.to_dict() for s in recent_submissions],
        'recent_violations': [v.to_dict() for v in recent_violations]
    })

@app.route('/api/all-submissions')
@login_required
def all_submissions():
    """Get all submissions"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    submissions = TestSubmission.query.order_by(
        TestSubmission.submitted_at.desc()
    ).all()
    
    return jsonify({
        'success': True,
        'submissions': [s.to_dict() for s in submissions]
    })

# ========================================
# SOCKETIO EVENTS
# ========================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")

@socketio.on('join_dashboard')
def handle_join_dashboard():
    """Admin joins dashboard room"""
    join_room('dashboard')
    emit('joined', {'message': 'Joined dashboard room'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)