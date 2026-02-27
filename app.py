# app.py - AI Recruiting System - Complete Backend (Gemini Edition)
from flask import Flask, request, jsonify, session, send_from_directory, send_file, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from config import Config
import cv2, base64, numpy as np, json, os, random, string, secrets, re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

# ── App Init ───────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
app.config.from_object(Config)
Config.init_app(app)

# Dynamically allow ngrok URL if set
_allowed_origins = ["http://localhost:5000", "http://127.0.0.1:5000"]
_base_url = os.environ.get("BASE_URL", "")
if _base_url and _base_url not in _allowed_origins:
    _allowed_origins.append(_base_url)
CORS(app, supports_credentials=True, origins=_allowed_origins)
db = SQLAlchemy(app)
_socketio_origins = ['http://localhost:5000','http://127.0.0.1:5000']
_sio_base = os.environ.get('BASE_URL','')
if _sio_base and _sio_base not in _socketio_origins:
    _socketio_origins.append(_sio_base)
socketio = SocketIO(app, cors_allowed_origins=_socketio_origins, async_mode='eventlet', allow_upgrades=True, ping_timeout=60, ping_interval=25)
login_manager = LoginManager(app)
login_manager.login_view = 'serve_index'
login_manager.session_protection = 'strong'

# ── Global Error Handlers (ensures API routes always return JSON, never HTML) ───
@app.errorhandler(500)
def handle_500(e):
    import traceback
    traceback.print_exc()
    return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Endpoint not found'}), 404
    return send_from_directory('.', 'index.html')

@app.errorhandler(401)
def handle_401(e):
    return jsonify({'success': False, 'message': 'Authentication required'}), 401

@app.errorhandler(403)
def handle_403(e):
    return jsonify({'success': False, 'message': 'Forbidden'}), 403

# ── Models ─────────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username         = db.Column(db.String(80),  unique=True, nullable=False, index=True)
    email            = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash    = db.Column(db.String(512), nullable=False)
    full_name        = db.Column(db.String(150))
    role             = db.Column(db.String(20), default='candidate')
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    smtp_email       = db.Column(db.String(120))   # recruiter's Gmail used for sending
    smtp_app_password= db.Column(db.String(256))   # Gmail App Password (stored as-is)

    violations  = db.relationship('Violation',      backref='user', lazy='dynamic', cascade='all,delete-orphan')
    submissions = db.relationship('TestSubmission', backref='user', lazy='dynamic', cascade='all,delete-orphan')

    @property
    def is_admin(self):
        return self.role in ('admin', 'recruiter')

    @property
    def is_recruiter(self):
        return self.role in ('admin', 'recruiter')

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw, method='scrypt')

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class ExamQuestion(db.Model):
    __tablename__ = 'exam_questions'
    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    job_role      = db.Column(db.String(100), nullable=False, index=True)
    question_text = db.Column(db.Text,        nullable=False)
    option_a      = db.Column(db.String(500), nullable=False)
    option_b      = db.Column(db.String(500), nullable=False)
    option_c      = db.Column(db.String(500), nullable=False)
    option_d      = db.Column(db.String(500), nullable=False)
    correct_answer= db.Column(db.String(1),   nullable=False)
    difficulty    = db.Column(db.String(20),  default='medium')
    category      = db.Column(db.String(100))

    def to_dict(self, include_answer=False):
        d = {
            'id': self.id, 'job_role': self.job_role,
            'question_text': self.question_text,
            'options': {'a': self.option_a, 'b': self.option_b,
                        'c': self.option_c, 'd': self.option_d},
            'difficulty': self.difficulty, 'category': self.category
        }
        if include_answer:
            d['correct_answer'] = self.correct_answer
        return d


class InterviewSession(db.Model):
    __tablename__ = 'interview_sessions'
    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    candidate_id  = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    recruiter_id  = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    job_role      = db.Column(db.String(100), nullable=False)
    mode          = db.Column(db.String(30),  nullable=False)
    room_code     = db.Column(db.String(16),  unique=True, nullable=False)
    status        = db.Column(db.String(20),  default='pending')
    credibility_score = db.Column(db.Integer, default=100)
    interview_score   = db.Column(db.Integer, default=0)
    started_at    = db.Column(db.DateTime)
    ended_at      = db.Column(db.DateTime)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    question_ids  = db.Column(db.Text)
    ai_transcript = db.Column(db.Text)
    recruiter_notes = db.Column(db.Text)

    candidate = db.relationship('User', foreign_keys=[candidate_id], backref='sessions_as_candidate')
    recruiter = db.relationship('User', foreign_keys=[recruiter_id], backref='sessions_as_recruiter')

    def to_dict(self):
        return {
            'id': self.id,
            'candidate_name': self.candidate.full_name or self.candidate.username,
            'candidate_username': self.candidate.username,
            'job_role': self.job_role,
            'mode': self.mode,
            'room_code': self.room_code,
            'status': self.status,
            'credibility_score': self.credibility_score,
            'interview_score': self.interview_score,
            'started_at': self.started_at.strftime('%Y-%m-%d %H:%M:%S') if self.started_at else None,
            'ended_at':   self.ended_at.strftime('%Y-%m-%d %H:%M:%S') if self.ended_at else None,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        }


class Violation(db.Model):
    __tablename__ = 'violations'
    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    session_id    = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True)
    violation_type= db.Column(db.String(50), nullable=False, index=True)
    timestamp     = db.Column(db.DateTime,   default=datetime.utcnow, index=True)
    severity      = db.Column(db.Integer,    default=1)
    description   = db.Column(db.Text)
    gaze_data     = db.Column(db.Text)
    device_data   = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id, 'user_id': self.user_id,
            'username': self.user.username,
            'violation_type': self.violation_type,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'severity': self.severity, 'description': self.description
        }


class GazeEvent(db.Model):
    __tablename__ = 'gaze_events'
    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True)
    direction  = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)


class DeviceAlert(db.Model):
    __tablename__ = 'device_alerts'
    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id  = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True)
    device_type = db.Column(db.String(50))
    confidence  = db.Column(db.Float)
    image_b64   = db.Column(db.Text)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)


class TestSubmission(db.Model):
    __tablename__ = 'test_submissions'
    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id      = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='SET NULL'), nullable=True)
    job_role        = db.Column(db.String(100), default='General')
    mode            = db.Column(db.String(30),  default='mcq')
    answers         = db.Column(db.Text, nullable=False)
    credibility_score   = db.Column(db.Integer, default=100)
    interview_score     = db.Column(db.Integer, default=0)
    total_violations    = db.Column(db.Integer, default=0)
    submitted_at    = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    exam_duration_seconds = db.Column(db.Integer)
    passed          = db.Column(db.Boolean, default=False)
    attempt_number  = db.Column(db.Integer, default=1)
    ai_feedback     = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id, 'user_id': self.user_id,
            'username': self.user.username,
            'full_name': self.user.full_name,
            'job_role': self.job_role or 'General',
            'mode': self.mode,
            'answers': json.loads(self.answers),
            'credibility_score': self.credibility_score,
            'interview_score': self.interview_score,
            'total_violations': self.total_violations,
            'submitted_at': self.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
            'exam_duration_seconds': self.exam_duration_seconds,
            'passed': self.passed,
            'attempt_number': self.attempt_number,
            'ai_feedback': self.ai_feedback,
        }


# ── Login Manager ──────────────────────────────────────────────────────────────
@login_manager.user_loader
def load_user(uid): return db.session.get(User, int(uid))

@login_manager.unauthorized_handler
def unauthorized():
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Login required'}), 401
    return send_from_directory('.', 'index.html')


# ── Helpers ────────────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def make_room_code(n=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def get_violation_info(vtype):
    return {
        'tab_switch':       {'severity': 2, 'description': 'User switched tab/window'},
        'exit_fullscreen':  {'severity': 3, 'description': 'Exited fullscreen'},
        'no_face':          {'severity': 2, 'description': 'Face not visible'},
        'multiple_faces':   {'severity': 3, 'description': 'Multiple faces detected'},
        'right_click':      {'severity': 1, 'description': 'Right-click attempt'},
        'copy_attempt':     {'severity': 1, 'description': 'Copy attempt'},
        'paste_attempt':    {'severity': 1, 'description': 'Paste attempt'},
        'devtools':         {'severity': 2, 'description': 'DevTools attempt'},
        'gaze_away':        {'severity': 2, 'description': 'Looking away from screen'},
        'phone_detected':   {'severity': 3, 'description': 'Phone/device detected'},
        'device_detected':  {'severity': 3, 'description': 'Unauthorized device detected'},
        'second_screen':    {'severity': 3, 'description': 'Second screen detected'},
    }.get(vtype, {'severity': 1, 'description': 'Unknown violation'})

def calc_credibility(session_id):
    violations = Violation.query.filter_by(session_id=session_id).all()
    score = 100
    for v in violations:
        score -= app.config['SEVERITY_POINTS'].get(v.severity, 5)
    if len(violations) > 10:
        score -= (len(violations) - 10) * 2
    return max(0, min(100, score))

def log_violation_db(user_id, session_id, vtype, gaze_data=None, device_data=None):
    info = get_violation_info(vtype)
    v = Violation(
        user_id=user_id, session_id=session_id,
        violation_type=vtype, severity=info['severity'],
        description=info['description'],
        gaze_data=json.dumps(gaze_data) if gaze_data else None,
        device_data=json.dumps(device_data) if device_data else None,
    )
    db.session.add(v)
    db.session.commit()
    sess = db.session.get(InterviewSession, session_id) if session_id else None
    new_score = calc_credibility(session_id) if session_id else 100
    if sess:
        sess.credibility_score = new_score
        db.session.commit()
    socketio.emit('violation_alert', {
        'user_id': user_id,
        'username': db.session.get(User, user_id).username,
        'violation_type': vtype, 'severity': info['severity'],
        'timestamp': datetime.utcnow().strftime('%H:%M:%S'),
        'new_credibility': new_score,
    }, room='dashboard')
    return v


# ── Gemini Helper ──────────────────────────────────────────────────────────────
def get_gemini_client():
    """
    Returns a configured Gemini GenerativeModel.
    Install SDK:  pip install google-generativeai
    Set key:      set GEMINI_API_KEY=AIza...your-key-here
    """
    import google.generativeai as genai
    api_key = app.config.get('GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')   # fast + free tier available

def gemini_ask(prompt):
    """Send a prompt to Gemini and return the text response. Returns None on failure."""
    model = get_gemini_client()
    if not model:
        return None
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

def parse_json_response(raw):
    """Strip markdown code fences and parse JSON safely."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith('```'):
        parts = text.split('```')
        # parts[1] is the content inside the first fence pair
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ── DB Seed ────────────────────────────────────────────────────────────────────
QUESTION_BANK = {
    "Python Developer": [
        ("What is a Python decorator?","A function that modifies another function's behavior","A design pattern for objects","A type of Python variable","A method to import modules","a","Core Python"),
        ("What does GIL stand for in Python?","General Interface Layer","Global Interpreter Lock","Generic Import Library","Global Instance List","b","Core Python"),
        ("Which library is used for data manipulation?","NumPy","Matplotlib","Pandas","SciPy","c","Libraries"),
        ("What does 'yield' do in Python?","Returns a value and exits","Pauses a function and returns a value (generator)","Imports an external library","Declares a global variable","b","Advanced Python"),
        ("What is __init__ in Python classes?","Destroys objects","Class constructor called on instantiation","A static method","A module initializer","b","OOP"),
        ("How do you handle exceptions in Python?","if/else blocks","try/except blocks","switch/case","catch/throw","b","Error Handling"),
        ("What is the difference between list and tuple?","Lists store more elements","Tuples are mutable, lists immutable","Lists are mutable, tuples immutable","No difference","c","Data Structures"),
        ("What is a virtual environment?","A remote server","Isolated Python env with its own packages","A GUI framework","A debugging tool","b","Development"),
        ("What does the 'with' statement do?","Creates a loop","Defines a function","Manages context and ensures resource cleanup","Imports modules","c","Core Python"),
        ("What is the purpose of *args?","Pass keyword arguments","Pass variable number of positional arguments","Define default values","Import arguments","b","Core Python"),
        ("What is duck typing?","Duck-shaped data structures","Type system based on object behavior not class","Automatic type conversion","Testing methodology","b","Advanced Python"),
        ("What does json.loads() do?","Saves data to JSON file","Converts JSON string to Python object","Loads a JSON module","Validates JSON syntax","b","Data Formats"),
        ("What is method resolution order (MRO)?","Order methods are defined","Order Python searches for methods in inheritance","Alphabetical method sort","Async method execution order","b","OOP"),
        ("What is list comprehension?","A way to sort lists","Concise way to create lists in one line","Merge multiple lists","Delete list elements","b","Core Python"),
        ("What does @property do in Python?","Makes a method static","Allows a method to be accessed like an attribute","Creates a private variable","Imports a property from module","b","OOP"),
    ],
    "Frontend Developer": [
        ("What is the Virtual DOM in React?","A real browser DOM","Lightweight in-memory copy of DOM for efficient updates","A virtual reality interface","A database for React state","b","React"),
        ("What is the purpose of async/await?","Make code run faster","Handle async operations in synchronous-like syntax","Create multiple threads","Delay function execution","b","JavaScript"),
        ("What does HTTP status 404 mean?","Server error","Unauthorized","Resource not found","Request timeout","c","Web Fundamentals"),
        ("What is event bubbling in JavaScript?","Floating animations","Event propagation from child to parent elements","Method for creating listeners","Canceling events","b","JavaScript"),
        ("What is the purpose of webpack?","CSS preprocessor","JavaScript testing framework","Module bundler that packages assets","Backend web framework","c","Tools"),
        ("What is CORS?","CSS layout system","Security mechanism for cross-origin HTTP requests","JavaScript testing library","React state management","b","Web Security"),
        ("Difference between == and === in JavaScript?","No difference","== checks value only, === checks value and type","=== checks value only","Both check type and value","b","JavaScript"),
        ("What is a Promise in JavaScript?","Guaranteed synchronous operation","Object for eventual completion of async operation","Way to create classes","A type of variable","b","JavaScript"),
        ("What is Flexbox used for?","Animation creation","Database connections","One-dimensional layout alignment","Creating 3D effects","c","CSS"),
        ("What is semantic HTML?","HTML with inline CSS","HTML tags that describe meaning of content","Minified HTML","HTML generated by JavaScript","b","HTML"),
        ("What is localStorage?","Server-side storage","Client-side key-value storage persisting across sessions","In-memory cache clearing on reload","Database for frontend","b","Web APIs"),
        ("What is the CSS Box Model?","3D rendering system","Layout model with margin, border, padding, content","Grid system for CSS","Flexbox config method","b","CSS"),
        ("What is a REST API?","Rapid execution system tool","Stateless web service using HTTP methods","Testing framework","Real-time protocol","b","APIs"),
        ("What is z-index in CSS?","Sets transparency","Controls stacking order of overlapping elements","Zooms in on elements","Sets position from top","b","CSS"),
        ("What does 'use strict' do in JavaScript?","Makes code faster","Enables strict mode catching common errors","Imports strict library","Disables debugging","b","JavaScript"),
    ],
    "Data Scientist": [
        ("What is overfitting?","Model performs well on train but poorly on unseen data","Model is too simple","Training takes too long","Dataset is too large","a","ML Concepts"),
        ("What is a confusion matrix?","A matrix that confuses the model","Table summarizing classification performance","Type of neural network layer","Data preprocessing technique","b","Model Evaluation"),
        ("What does SQL GROUP BY do?","Sorts alphabetically","Aggregates rows with same values in specified columns","Joins tables","Filters rows","b","SQL"),
        ("What is gradient descent?","Data visualization technique","Optimization algorithm minimizing a loss function","Type of decision tree","Method to normalize gradients","b","ML Algorithms"),
        ("What is a p-value?","Probability H0 is true","Probability of results as extreme assuming H0 is true","Mean of a distribution","Standard deviation of sample","b","Statistics"),
        ("Supervised vs unsupervised learning?","No difference","Supervised uses labeled data; unsupervised finds patterns in unlabeled","Supervised is faster","Unsupervised needs more data","b","ML Concepts"),
        ("What is a Random Forest?","Single decision tree","Ensemble of decision trees on random subsets","Neural network with random weights","Clustering algorithm","b","ML Algorithms"),
        ("What is feature engineering?","Building new models","Creating/transforming features to improve model performance","Removing irrelevant features","Visualizing features","b","Data Preparation"),
        ("What is cross-validation?","Combine multiple datasets","Assess model generalization on different data splits","Cross-check feature importance","Data augmentation technique","b","Model Evaluation"),
        ("What is regularization?","Makes models more complex","Penalty term to prevent overfitting","Data normalization method","Regularize training epochs","b","ML Concepts"),
        ("What is a hyperparameter?","Learned from training data","Config variable set before training, not learned from data","High-value feature","Parameter in test dataset","b","ML Concepts"),
        ("What is the Central Limit Theorem?","Mean equals median","Sample means approximate normal distribution as n increases","Large datasets are always normal","Central values are most important","b","Statistics"),
        ("What is dimensionality reduction?","Increasing features","Reducing features while preserving important information","Removing rows","Normalizing values","b","ML Concepts"),
        ("What does Pandas groupby() return?","List of DataFrames","GroupBy object for applying aggregate functions","Sorted DataFrame","Filtered dataset","b","Pandas"),
        ("What is PCA used for?","Predicting values","Reducing dimensions by finding principal components","Classifying data","Cleaning missing values","b","ML Algorithms"),
    ],
    "Backend Developer": [
        ("What is database indexing?","Numbering rows sequentially","Data structure speeding up data retrieval","Method to backup databases","Encrypting columns","b","Databases"),
        ("What is a foreign key?","Key to encrypt foreign data","Field referencing primary key in another table","Key from external API","Unique key for each row","b","Databases"),
        ("SQL vs NoSQL databases?","No difference","SQL uses structured schemas; NoSQL is flexible schema-less","NoSQL is always faster","SQL only for small datasets","b","Databases"),
        ("What is JWT?","JavaScript Web Toolkit","JSON Web Token for secure information transmission","Java Web Technology","JSON Write Tool","b","Security"),
        ("What is caching?","Permanently storing data","Temporarily storing data for faster future access","Deleting old records","Compressing database","b","Performance"),
        ("What is a microservices architecture?","One large application","Small independent services communicating via APIs","A database architecture","A frontend pattern","b","Architecture"),
        ("What is Docker?","A programming language","Platform to build, run, ship applications in containers","A database system","A testing framework","b","DevOps"),
        ("What is an ORM?","A REST framework","Object-Relational Mapper bridging code and database","A caching system","A message queue","b","Databases"),
        ("What is rate limiting?","Limiting data storage","Controlling request frequency to protect APIs from abuse","Limiting user accounts","A database constraint","b","API Design"),
        ("What is a message queue?","A database type","Asynchronous communication system between services","A logging system","A load balancer","b","Architecture"),
        ("What is ACID in databases?","A database query language","Atomicity, Consistency, Isolation, Durability – transaction properties","A NoSQL database","A backup strategy","b","Databases"),
        ("What is a RESTful API?","API using only GET","Architectural style using HTTP methods and stateless communication","A real-time socket API","API using only JSON","b","API Design"),
        ("What is load balancing?","Balancing database load","Distributing traffic across multiple servers","A caching technique","A security protocol","b","Architecture"),
        ("What is SQL injection?","A database optimization","Attack inserting malicious SQL code through user input","A query optimization","A join technique","b","Security"),
        ("What is a webhook?","A database trigger","HTTP callback sent when an event occurs in a system","A testing endpoint","A logging mechanism","b","API Design"),
    ],
    "Full Stack Developer": [
        ("What is the MVC pattern?","Model View Component","Model View Controller – separates concerns in an app","Multiple View Controller","Master Version Control","b","Architecture"),
        ("What is GraphQL?","A graph database","Query language for APIs allowing specific data requests","A JavaScript framework","A CSS preprocessor","b","APIs"),
        ("What is server-side rendering (SSR)?","Rendering on client browser","Rendering HTML on the server before sending to browser","A caching technique","A database pattern","b","Web Concepts"),
        ("What is a CDN?","Code Deployment Network","Content Delivery Network serving assets from nearest servers","Central Database Node","Code Design Notation","b","Infrastructure"),
        ("What is the purpose of environment variables?","Store user preferences","Store config values outside code, like API keys","Define global CSS","Manage database schemas","b","Development"),
        ("What is CI/CD?","Code Integration/Code Deployment","Continuous Integration/Continuous Deployment automation pipeline","A testing methodology","A version control system","b","DevOps"),
        ("What is HTTPS?","Hyper Text Transfer Protocol","Secure HTTP using TLS/SSL encryption","High Transfer Protocol Suite","Hybrid Transfer Protocol","b","Web Security"),
        ("What is session management?","Managing CSS sessions","Tracking user state across multiple HTTP requests","A database management pattern","A React concept","b","Web Concepts"),
        ("What is a monorepo?","A single-page app","Single repository containing multiple related projects","A database architecture","A deployment strategy","b","Development"),
        ("What is TypeScript?","A database language","Strongly typed superset of JavaScript","A CSS framework","A backend framework","b","Languages"),
        ("What is lazy loading?","Loading everything upfront","Deferring loading of resources until they are needed","A database technique","An animation effect","b","Performance"),
        ("What is OAuth?","A database protocol","Open authorization standard for delegated access","A CSS framework","A JavaScript engine","b","Security"),
        ("What is a proxy server?","A database server","Intermediary server between client and destination server","A CDN node","A caching database","b","Infrastructure"),
        ("What is Web Accessibility (a11y)?","Making websites faster","Designing websites usable by people with disabilities","A testing framework","An API standard","b","Web Standards"),
        ("What is the purpose of package.json?","Stores CSS","Defines project metadata, dependencies, and scripts for Node.js","Configures databases","Stores environment variables","b","Development"),
    ],
    "DevOps Engineer": [
        ("What is Infrastructure as Code (IaC)?","Manual server setup","Managing infrastructure through config files and code","A deployment strategy","A monitoring approach","b","IaC"),
        ("What is Kubernetes used for?","Code versioning","Orchestrating and managing containerized applications at scale","A CI/CD pipeline","A cloud database","b","Containers"),
        ("What is a Dockerfile?","A database config file","Script with instructions to build a Docker image","A Kubernetes config","A CI/CD config file","b","Docker"),
        ("What is blue-green deployment?","A CSS color theme","Deployment strategy with two identical environments to reduce downtime","A Kubernetes pod type","A monitoring strategy","b","Deployment"),
        ("What is Terraform?","A cloud provider","IaC tool to provision infrastructure across cloud providers","A containerization platform","A monitoring tool","b","IaC"),
        ("What does a reverse proxy do?","Forwards client requests to itself","Sits in front of servers forwarding client requests to them","Blocks incoming requests","Caches database queries","b","Infrastructure"),
        ("What is a CI pipeline?","Continuous Improvement","Automated process to build, test, and validate code changes","Code Integration Protocol","A deployment strategy","b","CI/CD"),
        ("What is observability in DevOps?","A monitoring tool","Ability to understand system state from its outputs (logs, metrics, traces)","A deployment pattern","An IaC concept","b","Monitoring"),
        ("What is Ansible used for?","Container orchestration","Automation tool for config management and app deployment","A monitoring platform","A CI/CD service","b","Automation"),
        ("What is a Helm chart?","A deployment diagram","Package manager for Kubernetes applications","A Docker config","A CI/CD pipeline","b","Kubernetes"),
        ("What is a service mesh?","A network type","Infrastructure layer managing service-to-service communication","A database pattern","A monitoring dashboard","b","Architecture"),
        ("What is GitOps?","Git best practices","Using Git as the single source of truth for infrastructure","A CI/CD tool","A code review process","b","DevOps"),
        ("What is log aggregation?","Deleting old logs","Collecting logs from multiple sources into a centralized system","A monitoring alert","A backup strategy","b","Monitoring"),
        ("What is autoscaling?","Fixed server count","Automatically adjusting compute resources based on demand","A load balancing method","A deployment pattern","b","Cloud"),
        ("What is a zero-downtime deployment?","Fast deployment","Deploying new code without interrupting service availability","A testing technique","A rollback strategy","b","Deployment"),
    ],
    "Mobile Developer": [
        ("What is the difference between native and hybrid mobile apps?","Native apps use HTML; hybrid use Swift","Native apps are built for specific platforms; hybrid use web tech wrapped in native container","Hybrid apps are faster than native","Native apps run in a browser","b","Mobile Basics"),
        ("What is React Native?","A native iOS framework","A JavaScript framework for building cross-platform mobile apps","A CSS framework for mobile","A database for mobile apps","b","Cross Platform"),
        ("What is the purpose of AndroidManifest.xml?","Stores app data","Declares app components, permissions, and metadata for Android","Configures the UI layout","Manages database connections","b","Android"),
        ("What is a ViewModel in Android?","A UI layout file","Stores and manages UI-related data surviving configuration changes","A database helper class","A network request class","b","Android Architecture"),
        ("What is Swift used for?","Android development","Building iOS and macOS applications","Cross-platform web apps","Backend development","b","iOS"),
        ("What is the difference between Activity and Fragment in Android?","No difference","Activity is a full screen; Fragment is a reusable portion of UI","Fragment is a full screen; Activity is reusable","Activities run in background","b","Android"),
        ("What is APK?","Apple Package Kit","Android Package Kit — the file format for Android app distribution","A programming language","A mobile database","b","Android"),
        ("What is Flutter?","A native Android framework","Google's UI toolkit for building cross-platform apps from a single codebase","An iOS testing tool","A mobile backend service","b","Cross Platform"),
        ("What is the purpose of a mobile app lifecycle?","To manage database connections","To manage app states like foreground, background, and terminated","To handle UI animations","To manage network requests","b","Mobile Basics"),
        ("What is push notification?","A UI gesture","Message sent from server to user device even when app is not open","A local notification","A database trigger","b","Mobile Features"),
        ("What is AsyncStorage in React Native?","A sync database","Simple unencrypted key-value storage system for React Native apps","A network storage","A cloud database","b","React Native"),
        ("What is Xcode?","A mobile framework","Apple's IDE for developing iOS and macOS applications","A cross-platform tool","A testing framework","b","iOS"),
        ("What is the purpose of Gradle in Android?","A UI framework","Build automation tool for Android projects managing dependencies","A testing tool","A version control system","b","Android"),
        ("What is deep linking in mobile apps?","A database connection","URL that navigates user directly to specific content inside an app","A network protocol","A security feature","b","Mobile Features"),
        ("What is mobile app signing?","Adding app logo","Process of digitally signing app to verify developer identity for distribution","Encrypting app data","Testing the app","b","Deployment"),
    ],
    "QA Engineer": [
        ("What is the difference between functional and non-functional testing?","No difference","Functional tests what system does; non-functional tests how system performs","Non-functional tests features; functional tests performance","Both test the same things","b","Testing Basics"),
        ("What is regression testing?","Testing new features only","Re-testing previously working features after code changes to ensure nothing broke","Testing performance under load","Testing security vulnerabilities","b","Testing Types"),
        ("What is a test case?","A bug report","A set of conditions and steps to verify a specific feature or behavior","A test environment","A testing tool","b","Testing Basics"),
        ("What is the difference between black box and white box testing?","Black box is manual; white box is automated","Black box tests without knowing internals; white box tests with knowledge of code","Black box is faster","White box is for UI testing only","b","Testing Types"),
        ("What is Selenium used for?","Mobile testing","Automating web browser interactions for testing","Load testing","API testing","b","Test Automation"),
        ("What is a bug life cycle?","How bugs are created","Stages a bug goes through from discovery to closure","How testers find bugs","The testing process","b","Bug Management"),
        ("What is smoke testing?","Testing for performance","Quick basic testing to check if build is stable enough for further testing","Testing all features thoroughly","Security testing","b","Testing Types"),
        ("What is the difference between verification and validation?","Same thing","Verification checks process; validation checks if product meets user needs","Validation checks code; verification checks UI","Both check requirements","b","Testing Concepts"),
        ("What is a test plan?","A list of bugs","Document describing testing scope, approach, resources and schedule","A test automation script","A bug tracking tool","b","Test Management"),
        ("What is API testing?","Testing mobile apps","Testing APIs directly to verify functionality, reliability and security","Testing user interface","Testing databases","b","Testing Types"),
        ("What is load testing?","Testing one user","Testing system behavior under expected and peak load conditions","Testing security","Testing UI layout","b","Performance Testing"),
        ("What is the purpose of a test environment?","Writing test cases","Isolated setup that mirrors production for safe testing without affecting live data","Storing bug reports","Running CI/CD pipelines","b","Testing Basics"),
        ("What is exploratory testing?","Automated testing","Simultaneous learning, test design and execution without predefined scripts","Performance testing","Unit testing","b","Testing Types"),
        ("What is JIRA used for in QA?","Writing code","Project management and bug tracking tool for managing issues and test cycles","Automated testing","Load testing","b","Tools"),
        ("What is a test automation framework?","A testing language","Structured guidelines and tools for creating and running automated tests efficiently","A bug tracking system","A test environment","b","Test Automation"),
    ],
    "Product Manager": [
        ("What is a product roadmap?","A list of bugs","Strategic plan showing product vision, direction and priorities over time","A project timeline","A marketing plan","b","Product Strategy"),
        ("What is the difference between product manager and project manager?","Same role","Product manager owns the what and why; project manager owns the how and when","Project manager owns the product vision","Product manager manages timelines","b","PM Basics"),
        ("What is a user story?","A bug report","Short description of a feature from the end user's perspective","A technical specification","A test case","b","Agile"),
        ("What is MVP in product management?","Most Valuable Player","Minimum Viable Product — simplest version to test core assumptions with users","Maximum Value Proposition","Minimum Value Proposition","b","Product Strategy"),
        ("What is product-market fit?","When product is launched","When product satisfies a strong market demand and users love it","When product has many features","When product is profitable","b","Product Strategy"),
        ("What are OKRs?","A project management tool","Objectives and Key Results — goal-setting framework for measurable outcomes","A design framework","A testing methodology","b","Goal Setting"),
        ("What is a sprint in Agile?","A long release cycle","Fixed time period (usually 2 weeks) where team completes a set of work","A bug fixing session","A design review","b","Agile"),
        ("What is the purpose of A/B testing?","Testing code quality","Comparing two versions of a feature to determine which performs better","Testing security","Load testing","b","Product Analytics"),
        ("What is the RICE scoring model?","A product design model","Prioritization framework using Reach, Impact, Confidence, Effort","A roadmap template","An agile ceremony","b","Prioritization"),
        ("What is a product backlog?","A list of completed features","Prioritized list of all desired features, improvements and bug fixes for a product","A sprint plan","A release schedule","b","Agile"),
        ("What is churn rate?","New user growth","Percentage of users who stop using product over a given period","Revenue growth rate","User engagement rate","b","Product Metrics"),
        ("What is the Jobs To Be Done framework?","A hiring framework","Understanding what problem or job users hire a product to do for them","A development framework","A testing methodology","b","Product Strategy"),
        ("What is DAU/MAU ratio?","A revenue metric","Daily Active Users divided by Monthly Active Users — measures user engagement stickiness","A bug tracking metric","A performance metric","b","Product Metrics"),
        ("What is a go-to-market strategy?","A product roadmap","Plan for launching product to market including target audience and channels","A development plan","A testing strategy","b","Product Strategy"),
        ("What is user persona?","A real user account","Semi-fictional representation of ideal customer based on research and data","A user story","A product feature","b","UX Research"),
    ],
    "UI/UX Designer": [
        ("What is the difference between UI and UX?","They are the same","UI is visual design of interface; UX is overall experience and usability","UX is visual; UI is experience","UI is for mobile; UX is for web","b","Design Basics"),
        ("What is a wireframe?","A final design","Low-fidelity skeletal outline of a screen showing layout without visual details","A prototype","A design system","b","Design Process"),
        ("What is a design system?","A single UI component","Collection of reusable components, guidelines and standards for consistent design","A wireframing tool","A prototyping method","b","Design Systems"),
        ("What is the purpose of user research?","To design UI","To understand user needs, behaviors and pain points to inform design decisions","To test performance","To write code","b","UX Research"),
        ("What is Figma used for?","Backend development","Collaborative UI/UX design tool for creating wireframes, prototypes and designs","Database management","Project management","b","Design Tools"),
        ("What is accessibility in design?","Making designs colorful","Designing products usable by people with disabilities following WCAG guidelines","Making designs responsive","Making designs fast","b","Accessibility"),
        ("What is a prototype?","A final product","Interactive simulation of product used to test and validate design ideas","A wireframe","A design system","b","Design Process"),
        ("What is the 80/20 rule in UX?","Design principle for colors","80% of users use 20% of features — focus on most impactful features","Rule for whitespace","Typography guideline","b","UX Principles"),
        ("What is information architecture?","Database structure","Organizing and structuring content so users can find information easily","UI component organization","Code architecture","b","UX Design"),
        ("What is usability testing?","Testing code","Observing real users interacting with product to identify usability issues","A/B testing","Performance testing","b","UX Research"),
        ("What is the difference between serif and sans-serif fonts?","No difference","Serif fonts have small strokes on letters; sans-serif fonts do not","Sans-serif is older","Serif is for digital only","b","Typography"),
        ("What is gestalt principle in design?","A color theory","Psychological principles explaining how humans perceive visual elements as unified wholes","A typography rule","A grid system","b","Design Principles"),
        ("What is responsive design?","A fast website","Design approach that makes UI adapt to different screen sizes and devices","A design tool","An animation technique","b","UI Design"),
        ("What is a user flow?","A design component","Visual representation of steps a user takes to complete a task in a product","A wireframe","A prototype","b","UX Design"),
        ("What is the purpose of white space in design?","Wasted space","Empty space around elements that improves readability and visual hierarchy","A design error","A color choice","b","Design Principles"),
    ],
    "Cybersecurity Analyst": [
        ("What is the CIA triad in cybersecurity?","Central Intelligence Agency model","Confidentiality, Integrity, Availability — core principles of information security","A network protocol","A hacking technique","b","Security Fundamentals"),
        ("What is a firewall?","A type of virus","Network security system that monitors and controls incoming and outgoing traffic","An encryption tool","A VPN service","b","Network Security"),
        ("What is phishing?","A network attack","Social engineering attack tricking users into revealing sensitive information","A malware type","A firewall bypass","b","Cyber Threats"),
        ("What is the difference between authentication and authorization?","Same thing","Authentication verifies identity; authorization determines what user can access","Authorization verifies identity","Authentication grants permissions","b","Access Control"),
        ("What is a VPN?","A firewall type","Virtual Private Network encrypting internet connection for secure private browsing","A type of malware","A network protocol","b","Network Security"),
        ("What is SQL injection?","A database optimization","Attack inserting malicious SQL code through input fields to manipulate database","A security protocol","A firewall rule","b","Web Security"),
        ("What is two-factor authentication?","A strong password","Security process requiring two different forms of verification to access account","A type of encryption","A firewall setting","b","Access Control"),
        ("What is a zero-day vulnerability?","An old bug","Security flaw unknown to vendor with no patch available yet","A firewall rule","A network attack","b","Vulnerabilities"),
        ("What is encryption?","Deleting data","Process of converting data into coded format to prevent unauthorized access","Compressing data","Backing up data","b","Cryptography"),
        ("What is penetration testing?","Breaking into buildings","Authorized simulated cyberattack to find and fix security vulnerabilities","A firewall test","A network scan","b","Security Testing"),
        ("What is a DDoS attack?","A phishing attack","Distributed Denial of Service — overwhelming server with traffic to make it unavailable","A SQL injection","A malware infection","b","Cyber Threats"),
        ("What is HTTPS?","A hacking tool","Secure version of HTTP using SSL/TLS encryption for safe data transmission","A firewall protocol","A VPN service","b","Web Security"),
        ("What is malware?","A security tool","Malicious software designed to damage, disrupt or gain unauthorized access to systems","A network protocol","A firewall type","b","Cyber Threats"),
        ("What is the principle of least privilege?","Give all users admin access","Grant users only minimum access permissions needed to perform their job","A firewall rule","An encryption method","b","Access Control"),
        ("What is a security audit?","A malware scan","Systematic evaluation of organization's security policies and controls for compliance","A penetration test","A firewall configuration","b","Security Management"),
    ],
    "Cloud Architect": [
        ("What are the three main cloud service models?","Public, Private, Hybrid","IaaS, PaaS, SaaS — Infrastructure, Platform and Software as a Service","AWS, Azure, GCP","Compute, Storage, Network","b","Cloud Fundamentals"),
        ("What is auto-scaling in cloud?","Manual server addition","Automatically adjusting compute resources up or down based on demand","A load balancer","A CDN service","b","Cloud Computing"),
        ("What is the difference between public and private cloud?","No difference","Public cloud is shared infrastructure; private cloud is dedicated to one organization","Private cloud is cheaper","Public cloud is more secure","b","Cloud Types"),
        ("What is a CDN?","A cloud database","Content Delivery Network distributing content from servers closest to users","A cloud storage service","A security service","b","Cloud Services"),
        ("What is serverless computing?","Computing without servers","Cloud model where provider manages servers and you only pay for actual execution","A type of virtual machine","A container service","b","Cloud Computing"),
        ("What is cloud elasticity?","Cloud flexibility in pricing","Ability to dynamically provision and de-provision resources as demand changes","A cloud security feature","A backup strategy","b","Cloud Concepts"),
        ("What is a cloud region?","A pricing zone","Geographic area containing multiple data centers for redundancy and latency optimization","A security boundary","A virtual network","b","Cloud Infrastructure"),
        ("What is object storage?","A database type","Storage architecture managing data as objects — used for unstructured data like images","A file system","A block storage type","b","Cloud Storage"),
        ("What is a virtual private cloud (VPC)?","A type of VPN","Isolated private network within public cloud infrastructure with full control over networking","A physical server","A cloud database","b","Cloud Networking"),
        ("What is cloud cost optimization?","Reducing cloud features","Strategies to reduce cloud spend while maintaining performance and reliability","Deleting cloud resources","Avoiding cloud services","b","Cloud Management"),
        ("What is multi-cloud strategy?","Using one cloud provider","Using multiple cloud providers to avoid vendor lock-in and improve resilience","A hybrid cloud approach","A private cloud setup","b","Cloud Strategy"),
        ("What is infrastructure as code?","Writing cloud documentation","Managing cloud infrastructure through machine-readable configuration files","A monitoring approach","A deployment strategy","b","Cloud Automation"),
        ("What is a load balancer in cloud?","A cost management tool","Distributes incoming traffic across multiple servers for high availability","A storage service","A database service","b","Cloud Networking"),
        ("What is disaster recovery in cloud?","Avoiding disasters","Strategy to restore systems and data after catastrophic failure or outage","A backup tool","A monitoring service","b","Cloud Reliability"),
        ("What is cloud monitoring?","A cloud database","Tracking performance, availability and health of cloud resources and applications","A security service","A cost management tool","b","Cloud Management"),
    ],
    "Database Administrator": [
        ("What is database normalization?","Making database faster","Organizing database to reduce redundancy and improve data integrity","Encrypting a database","Backing up a database","b","Database Design"),
        ("What is the difference between clustered and non-clustered index?","No difference","Clustered index sorts data rows physically; non-clustered creates separate structure","Non-clustered is faster","Clustered index is always better","b","Indexing"),
        ("What is a stored procedure?","A database backup","Precompiled SQL code stored in database and executed on demand","A database trigger","A view","b","SQL"),
        ("What is database replication?","Making database larger","Copying and synchronizing data across multiple database servers for redundancy","Encrypting database","Indexing data","b","Database Administration"),
        ("What is ACID in databases?","A cleaning solution","Atomicity, Consistency, Isolation, Durability — properties ensuring reliable transactions","A query language","A backup strategy","b","Database Concepts"),
        ("What is a database trigger?","A stored procedure","Automatic action executed in response to specific events on a table","A database index","A view","b","SQL"),
        ("What is the difference between DELETE and TRUNCATE?","Same command","DELETE removes specific rows with WHERE clause; TRUNCATE removes all rows faster","TRUNCATE is slower","DELETE removes all rows","b","SQL"),
        ("What is database sharding?","A backup technique","Horizontal partitioning of data across multiple database instances for scalability","A replication method","An indexing strategy","b","Scalability"),
        ("What is a database view?","A stored procedure","Virtual table based on result of SQL query providing simplified data access","A physical table","A backup copy","b","SQL"),
        ("What is query optimization?","Writing longer queries","Process of improving SQL query performance by reducing execution time and resources","Adding more indexes","Rewriting in another language","b","Performance"),
        ("What is a connection pool?","A database backup","Cache of database connections maintained for reuse to improve performance","A replication method","A security feature","b","Performance"),
        ("What is the difference between INNER JOIN and LEFT JOIN?","No difference","INNER JOIN returns matching rows; LEFT JOIN returns all left table rows plus matches","LEFT JOIN is faster","INNER JOIN returns all rows","b","SQL"),
        ("What is database backup strategy?","Deleting old data","Plan for regularly copying database to protect against data loss","A replication method","An indexing strategy","b","Database Administration"),
        ("What is a primary key?","Any column in a table","Unique identifier for each row in a table that cannot be null or duplicate","A foreign key","An index","b","Database Design"),
        ("What is NoSQL and when would you use it?","A bad database","Non-relational database used for unstructured data, high scalability and flexible schemas","A type of SQL","An old database type","b","Database Types"),
    ],
    "Machine Learning Engineer": [
        ("What is the difference between supervised and unsupervised learning?","No difference","Supervised uses labeled data; unsupervised finds patterns in unlabeled data","Supervised is faster","Unsupervised needs more data","b","ML Fundamentals"),
        ("What is overfitting and how do you prevent it?","Model is too simple","Model memorizes training data but fails on new data — prevented by regularization and more data","Model trains too slowly","Model has too many features","b","ML Concepts"),
        ("What is a neural network?","A biological brain","Computing system inspired by brain with layers of nodes learning patterns from data","A decision tree","A clustering algorithm","b","Deep Learning"),
        ("What is backpropagation?","Forward pass in neural network","Algorithm for training neural networks by computing gradients and updating weights","A regularization technique","A data preprocessing step","b","Deep Learning"),
        ("What is the purpose of a loss function?","To speed up training","Measures difference between model predictions and actual values to guide training","To normalize data","To split data","b","ML Training"),
        ("What is transfer learning?","Training from scratch","Using pretrained model on new related task to save time and data requirements","A regularization method","A data augmentation technique","b","Deep Learning"),
        ("What is the difference between batch and stochastic gradient descent?","Same algorithm","Batch uses all data per update; stochastic uses one sample — mini-batch is a compromise","Batch is always better","Stochastic is more accurate","b","ML Optimization"),
        ("What is feature scaling and why is it important?","Removing features","Normalizing feature ranges so no single feature dominates distance-based algorithms","Adding new features","Selecting important features","b","Data Preprocessing"),
        ("What is cross-validation?","Testing on training data","Technique to assess model generalization by training and testing on different data splits","A regularization method","A feature selection technique","b","Model Evaluation"),
        ("What is the ROC curve?","A training curve","Graph showing true positive rate vs false positive rate at various classification thresholds","A loss curve","A learning rate curve","b","Model Evaluation"),
        ("What is the difference between precision and recall?","Same metric","Precision measures correct positive predictions; recall measures coverage of actual positives","Precision is better than recall","Recall measures accuracy","b","Model Evaluation"),
        ("What is an embedding in machine learning?","A data format","Dense vector representation of data like words or categories in continuous space","A type of neural network","A preprocessing step","b","Deep Learning"),
        ("What is dropout in neural networks?","Removing neurons permanently","Regularization technique randomly disabling neurons during training to prevent overfitting","A type of activation function","A weight initialization method","b","Deep Learning"),
        ("What is hyperparameter tuning?","Training the model","Process of finding optimal model configuration values not learned from training data","Feature engineering","Data preprocessing","b","ML Optimization"),
        ("What is the bias-variance tradeoff?","A data imbalance issue","Balance between model being too simple (high bias) and too complex (high variance)","A regularization method","A feature selection technique","b","ML Concepts"),
    ],
    "Business Analyst": [
        ("What is a Business Requirements Document (BRD)?","A project plan","Formal document describing business solution needed including objectives and scope","A technical specification","A test plan","b","Requirements"),
        ("What is the difference between functional and non-functional requirements?","Same thing","Functional requirements describe what system does; non-functional describe how well it performs","Non-functional describes features","Functional describes performance","b","Requirements"),
        ("What is a use case diagram?","A project timeline","Visual representation showing interactions between users and system to achieve goals","A database diagram","A wireframe","b","UML"),
        ("What is gap analysis?","A performance review","Comparing current state to desired future state to identify what needs to change","A risk assessment","A requirements document","b","Analysis Techniques"),
        ("What is SWOT analysis?","A financial model","Framework analyzing Strengths, Weaknesses, Opportunities, Threats of a business","A project plan","A requirements method","b","Business Analysis"),
        ("What is a stakeholder?","A project manager","Anyone who has interest in or is affected by the outcome of a project","A developer","A business analyst","b","Project Management"),
        ("What is the purpose of a feasibility study?","Writing requirements","Assessing if proposed solution is technically and financially viable before development","A project plan","A test plan","b","Analysis Techniques"),
        ("What is process mapping?","A project roadmap","Visual representation of workflow steps showing how a business process works","A requirements document","A data flow diagram","b","Business Process"),
        ("What is an ERD?","A project diagram","Entity Relationship Diagram showing relationships between data entities in a system","A process map","A use case diagram","b","Data Modeling"),
        ("What is the MoSCoW method?","A city in Russia","Prioritization technique: Must have, Should have, Could have, Won't have","A requirements template","An analysis framework","b","Prioritization"),
        ("What is a data flow diagram?","A database schema","Visual showing how data moves through a system between processes and storage","A process map","An ERD","b","System Analysis"),
        ("What is the purpose of a sprint review?","Code review","Meeting at end of sprint where team demonstrates completed work to stakeholders","A bug review","A requirements review","b","Agile"),
        ("What is root cause analysis?","Finding bugs","Technique to identify underlying cause of a problem rather than treating symptoms","A risk assessment","A requirements review","b","Problem Solving"),
        ("What is a KPI?","A project tool","Key Performance Indicator — measurable value showing how effectively objectives are achieved","A requirements document","A test metric","b","Business Metrics"),
        ("What is change management in BA?","Managing code changes","Structured approach to transitioning individuals and organizations to desired future state","A version control process","A database update process","b","Business Analysis"),
    ],
}

def seed_questions():
    print("🌱 Checking question bank...")
    count = 0
    for role, questions in QUESTION_BANK.items():
        # Only seed roles that don't have questions yet
        existing = ExamQuestion.query.filter_by(job_role=role).first()
        if existing:
            continue
        print(f"   Adding questions for: {role}")
        for q in questions:
            eq = ExamQuestion(
                job_role=role, question_text=q[0],
                option_a=q[1], option_b=q[2],
                option_c=q[3], option_d=q[4],
                correct_answer=q[5],
                category=q[6] if len(q) > 6 else 'General'
            )
            db.session.add(eq)
            count += 1
    if count > 0:
        db.session.commit()
        print(f"✅ Seeded {count} new questions across {len(QUESTION_BANK)} roles")
    else:
        print(f"✅ Question bank already complete ({len(QUESTION_BANK)} roles)")

def init_database():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            u = User(username='admin', email='admin@proctoring.com', full_name='Administrator', role='admin')
            u.set_password('admin123'); db.session.add(u); db.session.commit()
        if not User.query.filter_by(username='recruiter').first():
            u = User(username='recruiter', email='recruiter@proctoring.com', full_name='Demo Recruiter', role='recruiter')
            u.set_password('recruiter123'); db.session.add(u); db.session.commit()
        if not User.query.filter_by(username='student').first():
            u = User(username='student', email='student@test.com', full_name='Test Student', role='candidate')
            u.set_password('student123'); db.session.add(u); db.session.commit()
        seed_questions()
        print("✅ Database ready")

with app.app_context():
    init_database()


# ── Static Routes ──────────────────────────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    # Return empty 204 so browser stops throwing 404 errors
    return '', 204

# Public pages
for page in ['index.html']:
    exec(f"""
@app.route('/{page}')
def serve_{page.replace('.','_').replace('-','_')}():
    return send_from_directory('.', '{page}')
""")

# Candidate-only pages
for page in ['test.html','dashboard.html','results.html','interview_room.html','interview_complete.html']:
    exec(f"""
@app.route('/{page}')
@login_required
def serve_{page.replace('.','_').replace('-','_')}():
    if current_user.role in ('admin','recruiter'):
        return redirect('/recruiter_dashboard.html')
    return send_from_directory('.', '{page}')
""")

# Recruiter-only pages
for page in ['recruiter_dashboard.html','recruiter_room.html']:
    exec(f"""
@app.route('/{page}')
@login_required
def serve_{page.replace('.','_').replace('-','_')}():
    if current_user.role not in ('admin','recruiter'):
        return redirect('/test.html')
    return send_from_directory('.', '{page}')
""")


# ── Auth Routes ────────────────────────────────────────────────────────────────
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username','').strip()
    password = data.get('password','')
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    logout_user()
    session.clear()
    login_user(user, remember=False)
    redirect = 'test.html' if user.role == 'candidate' else 'recruiter_dashboard.html'
    return jsonify({'success': True, 'redirect': redirect,
                    'role': user.role, 'full_name': user.full_name})


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username  = data.get('username','').strip()
    email     = data.get('email','').strip()
    password  = data.get('password','')
    full_name = data.get('full_name','').strip()
    role      = data.get('role','candidate')
    if not all([username, email, password, full_name]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
    # Validate email format
    email_regex = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
    if not email_regex.match(email):
        return jsonify({'success': False, 'message': 'Please enter a valid email address'}), 400
    # Validate email domain has valid TLD (basic check)
    domain_part = email.split('@')[1]
    tld = domain_part.split('.')[-1].lower()
    invalid_tlds = {'invalid','test','localhost','fake','example','local'}
    if tld in invalid_tlds or len(tld) < 2:
        return jsonify({'success': False, 'message': 'Email domain appears invalid. Please use a real email.'}), 400
    # Check for disposable/known-fake domains
    disposable_domains = {'mailinator.com','guerrillamail.com','temp-mail.org','throwaway.email',
        'fakeinbox.com','sharklasers.com','guerrillamailblock.com','grr.la','spam4.me',
        'trashmail.com','yopmail.com','mailnull.com','spamgourmet.com','tempinbox.com'}
    if domain_part.lower() in disposable_domains:
        return jsonify({'success': False, 'message': 'Disposable email addresses are not allowed. Please use your real email.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered. Please login.'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already taken'}), 400
    u = User(username=username, email=email, full_name=full_name, role=role)
    u.set_password(password)
    db.session.add(u); db.session.commit()
    return jsonify({'success': True, 'message': 'Account created! Please login.'})


@app.route('/logout')
@login_required
def logout():
    # End any active interview sessions for this user on logout
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


@app.route('/api/check-auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'username': current_user.username,
                        'full_name': current_user.full_name, 'role': current_user.role,
                        'is_admin': current_user.is_admin,
                        'is_recruiter': current_user.is_recruiter})
    return jsonify({'authenticated': False})


# ── Job Roles & Questions ──────────────────────────────────────────────────────
@app.route('/api/job-roles')
def get_job_roles():
    db_roles = db.session.query(ExamQuestion.job_role).distinct().all()
    roles = [r[0] for r in db_roles]
    for r in app.config.get('JOB_ROLES', []):
        if r not in roles:
            roles.append(r)
    return jsonify({'success': True, 'roles': sorted(roles)})


@app.route('/api/mcq-questions/<job_role>')
@login_required
def get_mcq_questions(job_role):
    questions = ExamQuestion.query.filter_by(job_role=job_role).all()
    if len(questions) < 10:
        questions = ExamQuestion.query.limit(10).all()
    selected = random.sample(questions, min(10, len(questions)))
    return jsonify({'success': True, 'questions': [q.to_dict() for q in selected], 'job_role': job_role})


# ── Session Management ─────────────────────────────────────────────────────────
@app.route('/api/create-session', methods=['POST'])
@login_required
def create_session():
    data = request.get_json() or {}
    job_role = data.get('job_role', 'Python Developer')
    mode     = data.get('mode', 'mcq')
    candidate_username = data.get('candidate_username')

    if current_user.role == 'candidate':
        candidate = current_user
        recruiter_id = None
    else:
        if candidate_username:
            candidate = User.query.filter_by(username=candidate_username).first()
            if not candidate:
                return jsonify({'success': False, 'message': 'Candidate not found'}), 404
        else:
            return jsonify({'success': False, 'message': 'candidate_username required'}), 400
        recruiter_id = current_user.id

    question_ids = []
    if mode == 'mcq':
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
    )
    db.session.add(sess); db.session.commit()
    return jsonify({'success': True, 'session': sess.to_dict(), 'room_code': sess.room_code})


@app.route('/api/join-session/<room_code>')
@login_required
def join_session(room_code):
    sess = InterviewSession.query.filter_by(room_code=room_code).first()
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found'}), 404

    # Guard 1: recruiters/admins can join as observers (for recruiter room)
    if current_user.is_admin:
        # Recruiter joining their own created session is allowed
        if sess.recruiter_id and sess.recruiter_id != current_user.id:
            return jsonify({'success': False,
                            'message': 'This session belongs to a different recruiter.'}), 403
        # Return session info for recruiter without candidate guards
        return jsonify({'success': True, 'session': sess.to_dict(),
                        'questions': [], 'ice_servers': app.config['WEBRTC_ICE_SERVERS']})

    # Guard 2: only the assigned candidate may join this session
    if sess.candidate_id != current_user.id:
        return jsonify({'success': False,
                        'message': 'This session was not assigned to your account.'}), 403

    # Guard 3: prevent rejoining a completed session
    if sess.status == 'completed':
        return jsonify({'success': False, 'message': 'already_submitted'}), 400

    questions = []
    if sess.mode == 'mcq' and sess.question_ids:
        ids = json.loads(sess.question_ids)
        questions = [db.session.get(ExamQuestion, i).to_dict() for i in ids
                     if db.session.get(ExamQuestion, i)]
    return jsonify({'success': True, 'session': sess.to_dict(),
                    'questions': questions, 'ice_servers': app.config['WEBRTC_ICE_SERVERS']})


@app.route('/api/start-session/<int:session_id>', methods=['POST'])
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


# ── Proctoring ─────────────────────────────────────────────────────────────────

@app.route('/api/room-scan', methods=['POST'])
@login_required
def room_scan():
    """Analyse 360 room scan frames for suspicious items/people."""
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        frames     = data.get('frames', [])
        if not frames:
            return jsonify({'success': True, 'issues': []})

        issues        = []
        multiple_seen = 0
        device_seen   = 0
        no_face_count = 0

        for idx, frame_b64 in enumerate(frames):
            try:
                raw   = frame_b64.split(',')[1] if ',' in frame_b64 else frame_b64
                buf   = np.frombuffer(base64.b64decode(raw), np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ── Face check ────────────────────────────────────────────────
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                nf = len(faces)
                if nf > 1:
                    multiple_seen += 1
                elif nf == 0 and idx < 2:
                    # Only flag no-face on first two frames (rest are scanning away)
                    no_face_count += 1

                # ── Phone/device check ────────────────────────────────────────
                h, w = frame.shape[:2]
                hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_b = np.array([100, 50, 50]); upper_b = np.array([130, 255, 255])
                mask = cv2.inRange(hsv, lower_b, upper_b)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 2000: continue
                    x2, y2, cw, ch = cv2.boundingRect(c)
                    ratio = float(max(cw, ch)) / float(min(cw, ch) + 1)
                    frame_ratio = area / float(w * h)
                    if 1.3 < ratio < 3.0 and frame_ratio < 0.35:
                        device_seen += 1

            except Exception as fe:
                print(f"Frame {idx} error: {fe}")
                continue

        # ── Determine issues ──────────────────────────────────────────────────
        if multiple_seen >= 2:
            issues.append("multiple people detected in room")
            log_violation_db(current_user.id, session_id, "multiple_faces",
                             gaze_data=f"Room scan: multiple people in {multiple_seen} frames")
        if device_seen >= 2:
            issues.append("prohibited device visible")
            log_violation_db(current_user.id, session_id, "device_detected",
                             device_data=f"Room scan: device in {device_seen} frames")
        if no_face_count >= 2:
            issues.append("candidate not visible at scan start")

        print(f"Room scan [{session_id}]: {len(frames)} frames, issues={issues}")
        return jsonify({'success': True, 'issues': issues,
                        'frames_analysed': len(frames)})

    except Exception as e:
        print(f"Room scan error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': True, 'issues': []})  # fail open — don't block interview

@app.route('/detect-face', methods=['POST'])
@login_required
def detect_face():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        img_b64    = data['image'].split(',')[1]
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # FIX: gentler scale factor (1.1) and larger minSize for more reliable detection
        faces      = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        n          = len(faces)

        # FIX: Only log violations if session is genuinely active AND past the 10-second
        # warm-up grace period. Previously violations fired immediately on join — before
        # camera was ready, during the rules page, and during the room scan — causing
        # candidates to start with 0 credibility before the interview even began.
        should_log = False
        if session_id:
            sess = db.session.get(InterviewSession, session_id)
            if sess and sess.status == 'active' and sess.started_at:
                elapsed = (datetime.utcnow() - sess.started_at).total_seconds()
                if elapsed >= 10:  # 10-second grace period after session starts
                    should_log = True

        if should_log:
            if n == 0:
                log_violation_db(current_user.id, session_id, 'no_face')
            elif n > 1:
                log_violation_db(current_user.id, session_id, 'multiple_faces')

        return jsonify({'success': True, 'face_detected': n == 1, 'num_faces': n})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze-gaze', methods=['POST'])
@login_required
def analyze_gaze():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        img_b64    = data['image'].split(',')[1]
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces      = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        gaze_result = {'direction': 'unknown', 'confidence': 0.0, 'looking_away': False}

        if len(faces) == 1:
            fx, fy, fw, fh = faces[0]
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 10, minSize=(20, 20))

            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]

                def pupil_center(roi_gray):
                    _, thresh = cv2.threshold(roi_gray, 70, 255, cv2.THRESH_BINARY_INV)
                    m = cv2.moments(thresh)
                    if m['m00'] != 0:
                        return int(m['m10']/m['m00']), int(m['m01']/m['m00'])
                    return roi_gray.shape[1]//2, roi_gray.shape[0]//2

                p1x, p1y = pupil_center(face_roi[ey1:ey1+eh1, ex1:ex1+ew1])
                p2x, p2y = pupil_center(face_roi[ey2:ey2+eh2, ex2:ex2+ew2])
                avg_ratio   = ((p1x/max(ew1,1)) + (p2x/max(ew2,1))) / 2
                avg_v_ratio = ((p1y/max(eh1,1)) + (p2y/max(eh2,1))) / 2

                if avg_ratio < 0.35:
                    direction, looking_away = 'left', True
                elif avg_ratio > 0.65:
                    direction, looking_away = 'right', True
                elif avg_v_ratio < 0.30:
                    direction, looking_away = 'up', True
                else:
                    direction, looking_away = 'center', False

                gaze_result = {'direction': direction, 'confidence': 0.80,
                               'looking_away': looking_away, 'ratio': round(avg_ratio, 3)}

                # FIX: same active+grace-period guard as detect-face
                should_log_gaze = False
                if session_id:
                    sess_g = db.session.get(InterviewSession, session_id)
                    if sess_g and sess_g.status == 'active' and sess_g.started_at:
                        if (datetime.utcnow() - sess_g.started_at).total_seconds() >= 10:
                            should_log_gaze = True

                if looking_away and should_log_gaze:
                    ge = GazeEvent(user_id=current_user.id, session_id=session_id,
                                   direction=direction, confidence=0.80)
                    db.session.add(ge); db.session.commit()
                    log_violation_db(current_user.id, session_id, 'gaze_away',
                                     gaze_data={'direction': direction, 'ratio': round(avg_ratio, 3)})
            elif len(eyes) == 0:
                gaze_result = {'direction': 'no_eyes', 'confidence': 0.5, 'looking_away': True}

        return jsonify({'success': True, 'gaze': gaze_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect-device', methods=['POST'])
@login_required
def detect_device():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        img_b64    = data['image'].split(',')[1]
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        fh, fw = frame.shape[:2]
        frame_area = fw * fh

        # Pre-processing
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq_gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(eq_gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges   = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Face regions to exclude
        eq_full  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces    = face_cascade.detectMultiScale(eq_full, 1.2, 5, minSize=(60, 60))
        face_regions = [(x, y, x+w, y+h) for (x, y, w, h) in faces]

        def in_face(cx, cy):
            return any(fx1 < cx < fx2 and fy1 < cy < fy2 for (fx1, fy1, fx2, fy2) in face_regions)

        # Bright-screen mask (screens emit bright, low-saturation light)
        hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        screen_mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        screen_cnts, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected    = False
        confidence  = 0.0
        device_type = None

        # ── Device shape profiles ──────────────────────────────────────────────
        # (label, min_area_ratio, max_area_ratio, min_aspect, max_aspect, base_conf)
        # Tightened aspect ratios & raised base confidence to cut false positives.
        # A phone is tall/narrow (aspect >= 1.6), tablets are squarish (1.2–1.9),
        # laptops are very wide (>= 2.2). This prevents walls/shirts being flagged.
        PROFILES = [
            ('phone',     0.006, 0.20,  1.7, 3.2,  0.82),
            ('tablet',    0.12,  0.40,  1.3, 1.8,  0.82),
            ('laptop',    0.15,  0.55,  2.4, 4.2,  0.81),
            ('earphones', 0.001, 0.008, 0.7, 1.5,  0.80),
        ]
        MIN_DETECT_CONF = 0.84  # raised — only flag very confident detections

        def check_rect(x, y, w, h, area):
            nonlocal detected, confidence, device_type
            aspect   = max(w, h) / max(min(w, h), 1)
            area_rat = area / frame_area
            cx, cy   = x + w // 2, y + h // 2
            if in_face(cx, cy):
                return
            for label, min_ar, max_ar, min_asp, max_asp, base_c in PROFILES:
                if min_ar <= area_rat <= max_ar and min_asp <= aspect <= max_asp:
                    fill = (area_rat - min_ar) / max(max_ar - min_ar, 1e-6)
                    conf = min(0.95, base_c + fill * 0.18)
                    if conf > confidence:
                        confidence, device_type = conf, label
                        detected = conf >= MIN_DETECT_CONF

        # Scan edge contours (shape-based)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frame_area * 0.0004: continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                check_rect(x, y, w, h, area)

        # Scan bright-screen contours — only flag if VERY screen-like
        # Raised min aspect to 2.0 and area threshold to avoid walls / white shirts
        for cnt in screen_cnts:
            area = cv2.contourArea(cnt)
            if area < frame_area * 0.025: continue   # was 0.012 — too small
            x, y, w, h = cv2.boundingRect(cnt)
            aspect   = max(w, h) / max(min(w, h), 1)
            area_rat = area / frame_area
            cx, cy   = x + w // 2, y + h // 2
            if in_face(cx, cy): continue
            # Only flag clearly rectangular, wide, moderately-sized screens
            # Aspect >= 1.4 avoids square walls; area cap 0.50 avoids filling the whole frame
            if 1.6 <= aspect <= 3.8 and 0.05 <= area_rat <= 0.45:
                label = 'laptop' if aspect >= 2.4 else 'tablet'
                conf  = min(0.92, 0.78 + area_rat * 0.20)
                if conf > confidence and conf >= MIN_DETECT_CONF:
                    confidence, device_type, detected = conf, label, True

        # ── Bluetooth / earpiece detection ─────────────────────────────────────
        # Improved: uses a larger ear zone (top 60% of frame), lower circularity
        # threshold (0.25 instead of 0.40) to catch bean-shaped earpieces,
        # and considers both contour shape and relative size to ear region.
        ear_zones = [(0, 0, fw//3, int(fh*0.65)), (fw*2//3, 0, fw, int(fh*0.65))]
        for (ex1, ey1, ex2, ey2) in ear_zones:
            roi   = blurred[ey1:ey2, ex1:ex2]
            # Use softer Canny edges so partially-hidden earpieces are found
            r_edg = cv2.Canny(roi, 20, 70)
            r_edg = cv2.dilate(r_edg, kernel, iterations=1)
            ecnts, _ = cv2.findContours(r_edg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in ecnts:
                area = cv2.contourArea(cnt)
                zone_area = max((ex2-ex1)*(ey2-ey1), 1)
                ea_ratio  = area / zone_area
                # Wider size window — small BT earpieces can be tiny in frame
                if not 0.003 <= ea_ratio <= 0.15: continue
                peri = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / max(peri ** 2, 1)
                aprx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                # Accept rounder shapes (circ > 0.25) OR compact polygons (4-8 sides)
                if circ > 0.25 or 4 <= len(aprx) <= 8:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cx2, cy2 = ex1+bx+bw//2, ey1+by+bh//2
                    if not in_face(cx2, cy2):
                        conf = min(0.88, 0.62 + ea_ratio * 1.8)
                        if conf > confidence and conf >= 0.65:
                            confidence, device_type, detected = conf, 'bluetooth_earpiece', True

        # FIX: same active+grace-period guard — don't log device violations during
        # the rules page, room scan, or camera warm-up phase.
        should_log_device = False
        if session_id:
            sess_d = db.session.get(InterviewSession, session_id)
            if sess_d and sess_d.status == 'active' and sess_d.started_at:
                if (datetime.utcnow() - sess_d.started_at).total_seconds() >= 10:
                    should_log_device = True

        if detected and should_log_device:
            _, buf = cv2.imencode('.jpg', cv2.resize(frame, (160, 120)))
            thumb  = base64.b64encode(buf).decode('utf-8')
            da = DeviceAlert(user_id=current_user.id, session_id=session_id,
                             device_type=device_type, confidence=confidence, image_b64=thumb)
            db.session.add(da); db.session.commit()
            log_violation_db(current_user.id, session_id, 'device_detected',
                             device_data={'device_type': device_type, 'confidence': round(confidence, 2)})

        return jsonify({'success': True, 'phone_detected': detected,
                        'device_type': device_type, 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/log-violation', methods=['POST'])
@login_required
def log_violation_route():
    data = request.get_json() or {}
    session_id = data.get('session_id')
    vtype = data.get('violation_type')

    # FIX: same active+grace-period guard for browser-side events (tab switch, right-click, copy)
    # These can fire during the rules page loading phase before the interview truly starts.
    if session_id:
        sess = db.session.get(InterviewSession, session_id)
        if not sess or sess.status != 'active' or not sess.started_at:
            return jsonify({'success': True, 'skipped': 'session not active yet'})
        elapsed = (datetime.utcnow() - sess.started_at).total_seconds()
        if elapsed < 5:  # 5-second grace for fast events like tab_switch
            return jsonify({'success': True, 'skipped': 'grace period'})

    log_violation_db(current_user.id, session_id, vtype)
    return jsonify({'success': True})


@app.route('/get-credibility-score')
@login_required
def get_credibility():
    session_id = request.args.get('session_id', type=int)
    score = calc_credibility(session_id) if session_id else 100
    v_count = Violation.query.filter_by(session_id=session_id).count() if session_id else 0
    return jsonify({'success': True, 'credibility_score': score, 'total_violations': v_count})


# ══════════════════════════════════════════════════════════════════════════════
# ── AI ROUTES — powered by Gemini (google-generativeai) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/ai-generate-questions', methods=['POST'])
@login_required
def ai_generate_questions():
    """
    Uses Gemini to generate 10 role-specific interview questions.
    Falls back to DB questions if GEMINI_API_KEY is not set.
    """
    data     = request.get_json() or {}
    job_role = data.get('job_role', 'Software Developer')

    # ── Try Gemini ──
    prompt = f"""You are an expert technical recruiter interviewing a candidate for a {job_role} position.

Generate exactly 10 interview questions. Return ONLY a valid JSON array like this:
[
  {{
    "id": 1,
    "question": "Your interview question here?",
    "type": "technical",
    "expected_keywords": ["keyword1", "keyword2", "keyword3"],
    "follow_up": "A follow-up question if they answer correctly?"
  }}
]

Rules:
- Mix of difficulty: 3 easy, 4 medium, 3 hard
- Cover fundamentals, problem-solving, best practices, real-world scenarios
- Questions must be specific to {job_role}
- Sound like a real human recruiter is asking
- Return ONLY the JSON array. No explanation, no markdown, no extra text."""

    raw = gemini_ask(prompt)
    questions = parse_json_response(raw)

    if questions and isinstance(questions, list) and len(questions) > 0:
        return jsonify({'success': True, 'source': 'gemini', 'questions': questions})

    # ── Fallback to DB if Gemini fails or key not set ──
    print("⚠️  Gemini unavailable — falling back to DB questions")
    qs = ExamQuestion.query.filter_by(job_role=job_role).limit(10).all()
    if not qs:
        qs = ExamQuestion.query.limit(10).all()
    return jsonify({'success': True, 'source': 'db_fallback',
                    'questions': [q.to_dict() for q in qs]})


@app.route('/api/ai-evaluate-answer', methods=['POST'])
@login_required
def ai_evaluate_answer():
    """
    Uses Gemini to score a candidate's answer (1-10) and give feedback.
    Falls back to keyword-matching if Gemini is unavailable.
    """
    data     = request.get_json() or {}
    question = data.get('question', '')
    answer   = data.get('answer', '')
    job_role = data.get('job_role', 'Developer')
    keywords = data.get('expected_keywords', [])

    # ── Try Gemini ──
    prompt = f"""You are evaluating a {job_role} interview answer.

Question: {question}
Candidate's Answer: {answer}
Expected Keywords/Concepts: {', '.join(keywords) if keywords else 'N/A'}

Evaluate strictly and fairly. Return ONLY this JSON object (no markdown, no explanation):
{{
  "score": <integer 1-10>,
  "feedback": "<exactly 2 sentences of constructive feedback>",
  "follow_up": "<one follow-up question if the answer was incomplete, or null if complete>",
  "strong_points": ["<what they got right>"],
  "missing_points": ["<what was missing>"]
}}"""

    raw = gemini_ask(prompt)
    result = parse_json_response(raw)

    if result and 'score' in result:
        return jsonify({'success': True, **result})

    # ── Fallback: keyword-matching score ──
    print("⚠️  Gemini unavailable — using keyword fallback scoring")
    matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
    score   = min(10, max(1, 4 + round((matches / max(len(keywords), 1)) * 6)))
    return jsonify({'success': True, 'score': score,
                    'feedback': f'Answer recorded. {matches} of {len(keywords)} key concepts mentioned.',
                    'follow_up': None, 'strong_points': [], 'missing_points': []})


@app.route('/api/ai-final-evaluation', methods=['POST'])
@login_required
def ai_final_evaluation():
    """
    Uses Gemini to generate a full evaluation report from the complete Q&A transcript.
    """
    data       = request.get_json() or {}
    transcript = data.get('transcript', [])
    job_role   = data.get('job_role', 'Developer')

    if not transcript:
        return jsonify({'success': True, 'overall_score': 50,
                        'summary': 'No transcript available.', 'recommendation': 'Review required.'})

    # Build readable Q&A text
    qa_text = '\n\n'.join([
        f"Q{i+1}: {t.get('question','')}\nAnswer: {t.get('answer','')}\nScore: {t.get('score','?')}/10"
        for i, t in enumerate(transcript)
    ])

    # ── Try Gemini ──
    prompt = f"""You are a senior hiring manager reviewing a completed {job_role} interview.

Full Q&A Transcript:
{qa_text}

Based on all answers, provide a final evaluation. Return ONLY this JSON (no markdown, no extra text):
{{
  "overall_score": <integer 0-100>,
  "technical_score": <integer 0-100>,
  "communication_score": <integer 0-100>,
  "summary": "<3-4 sentence overall assessment of the candidate>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<area to improve 1>", "<area to improve 2>"],
  "recommendation": "<exactly one of: Strong Hire | Hire | Maybe | No Hire>"
}}"""

    raw = gemini_ask(prompt)
    result = parse_json_response(raw)

    if result and 'overall_score' in result:
        return jsonify({'success': True, **result})

    # ── Fallback: average the per-question scores ──
    print("⚠️  Gemini unavailable — using average score fallback")
    avg = sum(t.get('score', 5) for t in transcript) / max(len(transcript), 1)
    overall = int(avg * 10)
    rec = 'Strong Hire' if overall >= 80 else 'Hire' if overall >= 65 else 'Maybe' if overall >= 50 else 'No Hire'
    return jsonify({'success': True, 'overall_score': overall,
                    'technical_score': overall, 'communication_score': overall,
                    'summary': f'Interview completed with an average score of {avg:.1f}/10.',
                    'strengths': ['Completed the interview'], 'improvements': ['More detail needed'],
                    'recommendation': rec})


# ── Submit ─────────────────────────────────────────────────────────────────────
@app.route('/submit-test', methods=['POST'])
@login_required
def submit_test():
    data       = request.get_json() or {}
    session_id = data.get('session_id')
    cred_score = calc_credibility(session_id) if session_id else 100

    # Always read mode and job_role from the DB session — never trust the client
    # This prevents a candidate from submitting under the wrong mode
    db_sess = db.session.get(InterviewSession, session_id) if session_id else None
    mode     = db_sess.mode     if db_sess else data.get('mode', 'mcq')
    job_role = db_sess.job_role if db_sess else data.get('job_role', 'General')

    interview_score = 0
    if mode == 'mcq':
        answers = data.get('answers', {})
        question_ids = []
        if session_id:
            sess = db.session.get(InterviewSession, session_id)
            if sess and sess.question_ids:
                question_ids = json.loads(sess.question_ids)
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

    passed      = cred_score >= 50 and interview_score >= 50
    attempt_num = TestSubmission.query.filter_by(user_id=current_user.id).count() + 1

    sub = TestSubmission(
        user_id=current_user.id, session_id=session_id,
        job_role=job_role, mode=mode,
        answers=json.dumps(data.get('answers', {})),
        credibility_score=cred_score, interview_score=interview_score,
        total_violations=Violation.query.filter_by(session_id=session_id).count() if session_id else 0,
        exam_duration_seconds=data.get('duration_seconds', 0),
        passed=passed, attempt_number=attempt_num,
        ai_feedback=json.dumps(data.get('ai_feedback', {})),
    )
    db.session.add(sub)

    if session_id:
        sess = db.session.get(InterviewSession, session_id)
        if sess:
            sess.status = 'completed'
            sess.ended_at = datetime.utcnow()
            sess.credibility_score = cred_score
            sess.interview_score   = interview_score
    db.session.commit()

    return jsonify({'success': True, 'submission_id': sub.id,
                    'credibility_score': cred_score, 'interview_score': interview_score,
                    'total_violations': sub.total_violations, 'passed': passed,
                    'redirect': f'results.html?id={sub.id}'})


# ── Results ────────────────────────────────────────────────────────────────────
@app.route('/api/results/<int:submission_id>')
@login_required
def get_results(submission_id):
    sub = db.session.get(TestSubmission, submission_id)
    if not sub:
        return jsonify({"success": False, "message": "Submission not found"}), 404
    if sub.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    violations    = Violation.query.filter_by(session_id=sub.session_id).all() if sub.session_id else []
    gaze_events   = GazeEvent.query.filter_by(session_id=sub.session_id).count() if sub.session_id else 0
    device_alerts = DeviceAlert.query.filter_by(session_id=sub.session_id).count() if sub.session_id else 0
    breakdown = {}
    for v in violations:
        breakdown[v.violation_type] = breakdown.get(v.violation_type, 0) + 1
    ai_feedback = {}
    if sub.ai_feedback:
        try: ai_feedback = json.loads(sub.ai_feedback)
        except: pass
    return jsonify({'success': True, 'submission': sub.to_dict(),
                    'violations': [v.to_dict() for v in violations],
                    'breakdown': breakdown, 'gaze_events': gaze_events,
                    'device_alerts': device_alerts, 'ai_feedback': ai_feedback})


@app.route('/api/retake-exam', methods=['POST'])
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
    db.session.add(sess); db.session.commit()
    return jsonify({'success': True, 'session': sess.to_dict(),
                    'room_code': sess.room_code,
                    'message': 'Fresh session created. Credibility starts at 100.'})


# ── Dashboard ──────────────────────────────────────────────────────────────────
@app.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    return jsonify({'success': True,
        'stats': {
            'total_candidates': User.query.filter_by(role='candidate').count(),
            'total_submissions': TestSubmission.query.count(),
            'total_violations': Violation.query.count(),
            'active_sessions': InterviewSession.query.filter_by(status='active').count(),
            'total_gaze_events': GazeEvent.query.count(),
            'total_device_alerts': DeviceAlert.query.count(),
        },
        'recent_submissions': [s.to_dict() for s in
            TestSubmission.query.order_by(TestSubmission.submitted_at.desc()).limit(20).all()],
        'recent_violations': [v.to_dict() for v in
            Violation.query.order_by(Violation.timestamp.desc()).limit(30).all()],
        'active_sessions_list': [s.to_dict() for s in
            InterviewSession.query.filter_by(status='active').all()],
    })


@app.route('/api/candidates')
@login_required
def get_candidates():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    candidates = User.query.filter_by(role='candidate').all()
    return jsonify({'success': True, 'candidates': [
        {'id': c.id, 'username': c.username, 'full_name': c.full_name,
         'email': c.email, 'submissions': c.submissions.count()} for c in candidates
    ]})


# ── SocketIO ───────────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print(f'Connected: {request.sid}')

# Track who is in each room
_room_members = {}

@socketio.on('join_room_event')
def on_join_room(data):
    room = data.get('room')
    role = data.get('role', 'candidate')
    join_room(room)

    # Track members
    if room not in _room_members:
        _room_members[room] = []
    _room_members[room] = [m for m in _room_members[room] if m['sid'] != request.sid]
    _room_members[room].append({'sid': request.sid, 'role': role})

    # Tell everyone someone joined
    emit('user_joined', {'role': role, 'sid': request.sid}, room=room)

    if role == 'recruiter':
        # Tell candidates recruiter arrived
        emit('recruiter_joined', {'room': room}, room=room, include_self=False)
        # If candidate already in room, tell recruiter immediately
        existing = [m for m in _room_members[room] if m['role'] == 'candidate']
        if existing:
            emit('candidate_present', {'room': room, 'count': len(existing)})
            print(f'Recruiter joined {room} - {len(existing)} candidate(s) already present')
    else:
        # If recruiter already in room, tell candidate
        recruiters = [m for m in _room_members[room] if m['role'] == 'recruiter']
        if recruiters:
            emit('recruiter_joined', {'room': room})
            print(f'Candidate joined {room} - recruiter already present')

@socketio.on('leave_room_event')
def on_leave_room(data):
    room = data.get('room')
    leave_room(room)
    if room in _room_members:
        _room_members[room] = [m for m in _room_members[room] if m['sid'] != request.sid]

@socketio.on('webrtc_offer')
def on_offer(data):
    emit('webrtc_offer', data, room=data['room'], include_self=False)

@socketio.on('webrtc_answer')
def on_answer(data):
    emit('webrtc_answer', data, room=data['room'], include_self=False)

@socketio.on('webrtc_ice_candidate')
def on_ice(data):
    emit('webrtc_ice_candidate', data, room=data['room'], include_self=False)

@socketio.on('recruiter_message')
def on_recruiter_msg(data):
    emit('recruiter_message', data, room=data['room'], include_self=False)

@socketio.on('end_interview')
def on_end_interview(data):
    emit('interview_ended', data, room=data['room'])

@socketio.on('join_dashboard')
def on_join_dashboard():
    join_room('dashboard')

@socketio.on('disconnect')
def on_disconnect():
    print(f'Disconnected: {request.sid}')
    # Clean up room membership
    for room in list(_room_members.keys()):
        _room_members[room] = [m for m in _room_members[room] if m["sid"] != request.sid]


# ── PDF Report ─────────────────────────────────────────────────────────────────
@app.route('/download-report/<int:submission_id>')
@login_required
def download_report(submission_id):
    sub = db.session.get(TestSubmission, submission_id)
    if not sub:
        return "Not Found", 404
    if sub.user_id != current_user.id and not current_user.is_admin:
        return "Unauthorized", 403
    buf = BytesIO()
    pdf = pdf_canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(1*inch, h-1*inch, "AI Recruiting System — Interview Report")
    pdf.setFont("Helvetica", 12)
    y = h - 1.6*inch
    for line in [f"Candidate: {sub.user.full_name} ({sub.user.username})",
                 f"Job Role: {sub.job_role}", f"Mode: {sub.mode.replace('_',' ').title()}",
                 f"Submitted: {sub.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Attempt: #{sub.attempt_number}"]:
        pdf.drawString(1*inch, y, line); y -= 0.28*inch
    y -= 0.2*inch
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(1*inch, y, f"Credibility Score:  {sub.credibility_score} / 100"); y -= 0.3*inch
    pdf.drawString(1*inch, y, f"Interview Score:    {sub.interview_score} / 100"); y -= 0.35*inch
    pdf.setFillColor(colors.green if sub.passed else colors.red)
    pdf.drawString(1*inch, y, f"Status: {'PASSED ✓' if sub.passed else 'FAILED ✗'}")
    pdf.setFillColor(colors.black)
    y -= 0.5*inch
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(1*inch, y, f"Total Violations: {sub.total_violations}")
    violations = Violation.query.filter_by(session_id=sub.session_id).all()
    y -= 0.3*inch
    pdf.setFont("Helvetica", 10)
    for v in violations[:20]:
        y -= 0.22*inch
        if y < 1*inch: pdf.showPage(); y = h - 1*inch
        pdf.drawString(
    1*inch,
    y,
    f"  • {v.timestamp.strftime('%H:%M:%S')}  {v.violation_type}  [{ {1:'Low', 2:'Medium', 3:'High'}.get(v.severity, '') }]"
)
    if sub.ai_feedback:
        try:
            fb = json.loads(sub.ai_feedback)
            y -= 0.5*inch
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(1*inch, y, "AI Interview Feedback:"); y -= 0.25*inch
            pdf.setFont("Helvetica", 10)
            summary = fb.get('summary', '')
            for chunk in [summary[i:i+90] for i in range(0, len(summary), 90)]:
                y -= 0.22*inch
                if y < 1*inch: pdf.showPage(); y = h - 1*inch
                pdf.drawString(1*inch, y, chunk)
        except: pass
    pdf.setFont("Helvetica", 8)
    pdf.drawString(1*inch, 0.5*inch, "Generated by AI Recruiting & Proctoring System (Gemini Edition)")
    pdf.save(); buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'report_{sub.user.username}_{sub.id}.pdf',
                     mimetype='application/pdf')




# ════════════════════════════════════════════════════════════════════════════════
# NEW MODELS — Job Postings, Applications, Scheduled Interviews
# ════════════════════════════════════════════════════════════════════════════════

class JobPosting(db.Model):
    __tablename__ = 'job_postings'
    id                  = db.Column(db.Integer, primary_key=True, autoincrement=True)
    recruiter_id        = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    job_section         = db.Column(db.String(100), nullable=False)   # e.g. IT, HR, Marketing
    job_role            = db.Column(db.String(100), nullable=False)
    job_title           = db.Column(db.String(200))                   # Custom title (e.g. Senior Python Dev)
    company_name        = db.Column(db.String(150), nullable=False)
    description         = db.Column(db.Text)
    skills_required     = db.Column(db.Text)                          # JSON array of skill strings
    experience_required = db.Column(db.String(50))                    # Fresher, Junior, Mid-Level, Senior, Lead
    job_type            = db.Column(db.String(50))                    # Full-Time, Part-Time, Contract, etc.
    is_active           = db.Column(db.Boolean, default=True)
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)

    recruiter     = db.relationship('User', foreign_keys=[recruiter_id])
    applications  = db.relationship('JobApplication', backref='posting', lazy='dynamic', cascade='all,delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'recruiter_id': self.recruiter_id,
            'recruiter_name': self.recruiter.full_name or self.recruiter.username,
            'job_section': self.job_section,
            'job_role': self.job_role,
            'job_title': self.job_title or self.job_role,
            'company_name': self.company_name,
            'description': self.description or '',
            'skills_required': json.loads(self.skills_required) if self.skills_required else [],
            'experience_required': self.experience_required or '',
            'job_type': self.job_type or '',
            'is_active': self.is_active,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'application_count': self.applications.count(),
        }


class JobApplication(db.Model):
    __tablename__ = 'job_applications'
    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    posting_id    = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='CASCADE'), nullable=False)
    candidate_id  = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    cover_note    = db.Column(db.Text)
    status        = db.Column(db.String(30), default='applied')  # applied | shortlisted | rejected | interview_scheduled
    applied_at    = db.Column(db.DateTime, default=datetime.utcnow)

    candidate     = db.relationship('User', foreign_keys=[candidate_id])

    def to_dict(self):
        return {
            'id': self.id,
            'posting_id': self.posting_id,
            'job_section': self.posting.job_section,
            'job_role': self.posting.job_role,
            'company_name': self.posting.company_name,
            'recruiter_name': self.posting.recruiter.full_name or self.posting.recruiter.username,
            'recruiter_email': self.posting.recruiter.email,
            'candidate_id': self.candidate_id,
            'candidate_name': self.candidate.full_name or self.candidate.username,
            'candidate_username': self.candidate.username,
            'candidate_email': self.candidate.email,
            'cover_note': self.cover_note or '',
            'status': self.status,
            'applied_at': self.applied_at.strftime('%Y-%m-%d %H:%M:%S'),
        }


class ScheduledInterview(db.Model):
    __tablename__ = 'scheduled_interviews'
    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    application_id  = db.Column(db.Integer, db.ForeignKey('job_applications.id', ondelete='CASCADE'), nullable=False)
    session_id      = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='SET NULL'), nullable=True)
    recruiter_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    candidate_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    scheduled_at    = db.Column(db.DateTime, nullable=False)
    interview_mode  = db.Column(db.String(30), default='mcq')
    calendar_link   = db.Column(db.Text)
    email_sent      = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    room_code       = db.Column(db.String(16))

    application = db.relationship('JobApplication', foreign_keys=[application_id])
    recruiter   = db.relationship('User', foreign_keys=[recruiter_id])
    candidate   = db.relationship('User', foreign_keys=[candidate_id])
    session     = db.relationship('InterviewSession', foreign_keys=[session_id])

    def to_dict(self):
        app = self.application
        return {
            'id': self.id,
            'application_id': self.application_id,
            'job_role': app.posting.job_role,
            'job_section': app.posting.job_section,
            'company_name': app.posting.company_name,
            'recruiter_name': self.recruiter.full_name or self.recruiter.username,
            'recruiter_email': self.recruiter.email,
            'candidate_name': self.candidate.full_name or self.candidate.username,
            'candidate_email': self.candidate.email,
            'scheduled_at': self.scheduled_at.strftime('%Y-%m-%d %H:%M:%S'),
            'interview_mode': self.interview_mode,
            'calendar_link': self.calendar_link or '',
            'email_sent': self.email_sent,
            'room_code': self.room_code or '',
            'session_id': self.session_id,
        }


# ── Rebuild DB to include new tables ──────────────────────────────────────────
with app.app_context():
    db.create_all()


# ── Email Helper ───────────────────────────────────────────────────────────────
def send_interview_email(to_email, to_name, recruiter_name, company_name,
                         job_role, scheduled_at_str, room_code, calendar_link,
                         role='candidate', smtp_user=None, smtp_pass=None):
    """
    Sends an interview notification email via SMTP.
    Uses recruiter's own Gmail credentials if provided,
    otherwise falls back to EMAIL_USER/EMAIL_PASS env vars.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_user = smtp_user or os.environ.get('EMAIL_USER', '')
    smtp_pass = smtp_pass or os.environ.get('EMAIL_PASS', '')

    subject = f"RecruitAI — Interview Scheduled: {job_role} at {company_name}"
    base_url = os.environ.get("BASE_URL", "http://localhost:5000")
    join_url = f"{base_url}/interview_room.html?room={room_code}"

    if role == 'candidate':
        body_html = f"""
<div style="font-family:Arial,sans-serif;max-width:580px;margin:0 auto;background:#f9fafb;padding:32px;border-radius:12px;">
  <div style="background:#0f1117;padding:24px;border-radius:10px;text-align:center;margin-bottom:24px;">
    <h1 style="color:#6aa3ff;margin:0;font-size:24px;">🤖 RecruitAI</h1>
    <p style="color:#7a88a8;margin:8px 0 0;">Interview Notification</p>
  </div>
  <h2 style="color:#1a2035;margin-bottom:8px;">Hello, {to_name}!</h2>
  <p style="color:#444;line-height:1.6;">Your interview has been <strong>scheduled</strong>. Here are the details:</p>
  <table style="width:100%;margin:20px 0;border-collapse:collapse;">
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;width:40%;border-radius:6px 6px 0 0;">Position</td><td style="padding:10px 14px;background:#fff;">{job_role}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Company</td><td style="padding:10px 14px;background:#fff;">{company_name}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Recruiter</td><td style="padding:10px 14px;background:#fff;">{recruiter_name}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Date &amp; Time</td><td style="padding:10px 14px;background:#fff;"><strong>{scheduled_at_str}</strong></td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Room Code</td><td style="padding:10px 14px;background:#fff;"><code style="background:#0f1117;color:#6aa3ff;padding:4px 8px;border-radius:4px;">{room_code}</code></td></tr>
  </table>
  <div style="text-align:center;margin:24px 0;">
    <a href="{join_url}" style="background:#4f8ef7;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px;">Join Interview →</a>
  </div>
  {f'<div style="text-align:center;margin:16px 0;"><a href="{calendar_link}" style="color:#4f8ef7;">📅 Add to Google Calendar</a></div>' if calendar_link else ''}
  <p style="color:#777;font-size:13px;margin-top:24px;">Please be on time and ensure you are in a quiet, well-lit environment. Any device violations will be monitored.</p>
  <p style="color:#aaa;font-size:12px;text-align:center;margin-top:32px;">RecruitAI — AI-Powered Recruitment Platform</p>
</div>
"""
    else:  # recruiter
        body_html = f"""
<div style="font-family:Arial,sans-serif;max-width:580px;margin:0 auto;background:#f9fafb;padding:32px;border-radius:12px;">
  <div style="background:#0f1117;padding:24px;border-radius:10px;text-align:center;margin-bottom:24px;">
    <h1 style="color:#6aa3ff;margin:0;font-size:24px;">🤖 RecruitAI</h1>
  </div>
  <h2 style="color:#1a2035;">Interview Scheduled — Recruiter Copy</h2>
  <p style="color:#444;">You have scheduled an interview. Room code: <code style="background:#0f1117;color:#6aa3ff;padding:4px 8px;border-radius:4px;">{room_code}</code></p>
  <table style="width:100%;margin:20px 0;border-collapse:collapse;">
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;width:40%;">Candidate</td><td style="padding:10px 14px;background:#fff;">{to_name}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Position</td><td style="padding:10px 14px;background:#fff;">{job_role}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Date &amp; Time</td><td style="padding:10px 14px;background:#fff;"><strong>{scheduled_at_str}</strong></td></tr>
  </table>
  {f'<div style="text-align:center;"><a href="{calendar_link}" style="color:#4f8ef7;">📅 Add to Google Calendar</a></div>' if calendar_link else ''}
</div>
"""

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = smtp_user or 'noreply@recruitai.com'
    msg['To']      = to_email
    msg.attach(MIMEText(body_html, 'html'))

    if smtp_user and smtp_pass:
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, [to_email], msg.as_string())
            print(f"✅ Email sent to {to_email}")
        except Exception as e:
            print(f"⚠️  Email failed: {e}")
    else:
        print(f"📧 [No SMTP] Would send to {to_email}: {subject}")

    return True


def make_google_calendar_link(title, description, start_dt, end_dt=None):
    """Generates a Google Calendar event creation link."""
    import urllib.parse
    if end_dt is None:
        from datetime import timedelta
        end_dt = start_dt + timedelta(hours=1)
    fmt = '%Y%m%dT%H%M%SZ'
    params = {
        'action': 'TEMPLATE',
        'text': title,
        'details': description,
        'dates': f"{start_dt.strftime(fmt)}/{end_dt.strftime(fmt)}",
    }
    return 'https://calendar.google.com/calendar/render?' + urllib.parse.urlencode(params)


# ════════════════════════════════════════════════════════════════════════════════
# JOB SECTION CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════
JOB_SECTIONS = [
    'IT', 'HR', 'Marketing', 'Finance', 'Sales', 'Operations',
    'Design', 'Product', 'Data & Analytics', 'Legal', 'Customer Support', 'Engineering'
]


@app.route('/api/job-sections')
def get_job_sections():
    return jsonify({'success': True, 'sections': JOB_SECTIONS})


# ════════════════════════════════════════════════════════════════════════════════
# JOB POSTINGS ROUTES (Recruiter)
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/job-postings', methods=['GET'])
@login_required
def list_job_postings():
    """List active job postings — candidates see all active, recruiters/admins see their own."""
    try:
        if current_user.is_admin or current_user.role == 'recruiter':
            postings = JobPosting.query.filter_by(recruiter_id=current_user.id, is_active=True).all()
        else:
            postings = JobPosting.query.filter_by(is_active=True).all()
        return jsonify({'success': True, 'postings': [p.to_dict() for p in postings]})
    except Exception as e:
        print(f"ERROR /api/job-postings: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e), 'postings': []}), 500


@app.route('/api/job-postings/all', methods=['GET'])
@login_required
def list_all_job_postings():
    """Recruiter/Admin: list all their postings including inactive."""
    if not (current_user.is_admin or current_user.role == 'recruiter'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        postings = JobPosting.query.filter_by(recruiter_id=current_user.id).order_by(JobPosting.created_at.desc()).all()
        return jsonify({'success': True, 'postings': [p.to_dict() for p in postings]})
    except Exception as e:
        print(f"ERROR /api/job-postings/all: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e), 'postings': []}), 500


@app.route('/api/job-postings', methods=['POST'])
@login_required
def create_job_posting():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    if not all([data.get('job_section'), data.get('job_role'), data.get('company_name')]):
        return jsonify({'success': False, 'message': 'job_section, job_role, and company_name are required'}), 400
    # Validate at least one skill
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
    )
    db.session.add(p); db.session.commit()
    return jsonify({'success': True, 'posting': p.to_dict()}), 201


@app.route('/api/job-postings/<int:posting_id>', methods=['DELETE'])
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


@app.route('/api/applicants-by-role', methods=['GET'])
@login_required
def get_applicants_by_role():
    """Return candidates who applied to any of this recruiter's postings for a given job_role."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    job_role = request.args.get('job_role', '').strip()
    if not job_role:
        return jsonify({'success': False, 'message': 'job_role is required'}), 400
    try:
        # Find all postings by this recruiter for that role
        postings = JobPosting.query.filter_by(recruiter_id=current_user.id, job_role=job_role, is_active=True).all()
        if not postings:
            return jsonify({'success': True, 'candidates': [], 'posting_ids': []})
        posting_ids = [p.id for p in postings]
        # Get all applications for those postings
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


@app.route('/api/job-postings/<int:posting_id>/applications', methods=['GET'])
@login_required
def get_applications_for_posting(posting_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    p = db.session.get(JobPosting, posting_id)
    if not p or p.recruiter_id != current_user.id:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    apps = JobApplication.query.filter_by(posting_id=posting_id).order_by(JobApplication.applied_at.desc()).all()
    return jsonify({'success': True, 'applications': [a.to_dict() for a in apps]})


# ════════════════════════════════════════════════════════════════════════════════
# JOB APPLICATION ROUTES (Candidate)
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/apply', methods=['POST'])
@login_required
def apply_for_job():
    if current_user.is_admin:
        return jsonify({'success': False, 'message': 'Recruiters cannot apply'}), 400
    data = request.get_json() or {}
    posting_id = data.get('posting_id')
    if not posting_id:
        return jsonify({'success': False, 'message': 'posting_id required'}), 400
    # Check if already applied
    existing = JobApplication.query.filter_by(posting_id=posting_id, candidate_id=current_user.id).first()
    if existing:
        return jsonify({'success': False, 'message': 'Already applied for this position'}), 400
    app_obj = JobApplication(
        posting_id=posting_id,
        candidate_id=current_user.id,
        cover_note=data.get('cover_note', ''),
        status='applied',
    )
    db.session.add(app_obj); db.session.commit()
    return jsonify({'success': True, 'application': app_obj.to_dict()})


@app.route('/api/my-applications')
@login_required
def my_applications():
    apps = JobApplication.query.filter_by(candidate_id=current_user.id).order_by(JobApplication.applied_at.desc()).all()
    # Also attach scheduled interview info
    result = []
    for a in apps:
        d = a.to_dict()
        sched = ScheduledInterview.query.filter_by(application_id=a.id).first()
        if sched:
            d['scheduled_interview'] = sched.to_dict()
        else:
            d['scheduled_interview'] = None
        result.append(d)
    return jsonify({'success': True, 'applications': result})


# ── Updated my-submissions to include application info ─────────────────────────
@app.route('/api/my-submissions')
@login_required
def my_submissions():
    subs = TestSubmission.query.filter_by(user_id=current_user.id).order_by(TestSubmission.submitted_at.desc()).all()
    return jsonify({'success': True, 'submissions': [s.to_dict() for s in subs]})


# ════════════════════════════════════════════════════════════════════════════════
# SCHEDULE INTERVIEW ROUTE (Recruiter)
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/schedule-interview', methods=['POST'])
@login_required
def schedule_interview():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    application_id = data.get('application_id')
    scheduled_at_str = data.get('scheduled_at')  # ISO format: 2025-03-15T14:30
    interview_mode = data.get('interview_mode', 'mcq')

    if not application_id or not scheduled_at_str:
        return jsonify({'success': False, 'message': 'application_id and scheduled_at required'}), 400

    app_obj = db.session.get(JobApplication, application_id)
    if not app_obj:
        return jsonify({'success': False, 'message': 'Application not found'}), 404

    # Parse scheduled time
    try:
        scheduled_dt = datetime.fromisoformat(scheduled_at_str)
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid datetime format'}), 400

    posting = app_obj.posting
    candidate = app_obj.candidate
    recruiter = current_user

    # Create an interview session
    question_ids = []
    if interview_mode == 'mcq':
        qs = ExamQuestion.query.filter_by(job_role=posting.job_role).all()
        if len(qs) < 10:
            qs = ExamQuestion.query.limit(15).all()
        selected = random.sample(qs, min(10, len(qs)))
        question_ids = [q.id for q in selected]

    room_code = make_room_code()
    sess = InterviewSession(
        candidate_id=candidate.id,
        recruiter_id=recruiter.id,
        job_role=posting.job_role,
        mode=interview_mode,
        room_code=room_code,
        status='pending',
        credibility_score=100,
        question_ids=json.dumps(question_ids),
    )
    db.session.add(sess); db.session.commit()

    # Build Google Calendar links
    title = f"RecruitAI Interview — {candidate.full_name or candidate.username} with {recruiter.full_name or recruiter.username}"
    desc = f"Job Role: {posting.job_role} at {posting.company_name}\nRoom Code: {room_code}\nJoin: {os.environ.get(chr(66)+chr(65)+chr(83)+chr(69)+chr(95)+chr(85)+chr(82)+chr(76), chr(104)+chr(116)+chr(116)+chr(112)+chr(58)+chr(47)+chr(47)+chr(108)+chr(111)+chr(99)+chr(97)+chr(108)+chr(104)+chr(111)+chr(115)+chr(116)+chr(58)+chr(53)+chr(48)+chr(48)+chr(48))}/interview_room.html?room={room_code}"
    cal_link = make_google_calendar_link(title, desc, scheduled_dt)

    # Create ScheduledInterview record
    sched = ScheduledInterview(
        application_id=application_id,
        session_id=sess.id,
        recruiter_id=recruiter.id,
        candidate_id=candidate.id,
        scheduled_at=scheduled_dt,
        interview_mode=interview_mode,
        calendar_link=cal_link,
        room_code=room_code,
    )
    db.session.add(sched)

    # Update application status
    app_obj.status = 'interview_scheduled'
    db.session.commit()

    scheduled_at_display = scheduled_dt.strftime('%B %d, %Y at %I:%M %p')

    # Send emails — use recruiter's own Gmail if configured, else fall back to env vars
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

    return jsonify({
        'success': True,
        'scheduled': sched.to_dict(),
        'room_code': room_code,
        'calendar_link': cal_link,
        'message': f'Interview scheduled for {scheduled_at_display}. Emails sent to both parties.'
    })


@app.route('/api/recruiter/email-settings', methods=['GET'])
@login_required
def get_email_settings():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    return jsonify({
        'success': True,
        'smtp_email': current_user.smtp_email or '',
        'has_password': bool(current_user.smtp_app_password),
    })

@app.route('/api/recruiter/email-settings', methods=['POST'])
@login_required
def save_email_settings():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json() or {}
    smtp_email = data.get('smtp_email', '').strip()
    smtp_pass  = data.get('smtp_app_password', '').strip()

    if smtp_email:
        # Basic email format check
        import re as _re
        if not _re.match(r'^[^@]+@[^@]+\.[^@]+$', smtp_email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400

    current_user.smtp_email = smtp_email or None
    if smtp_pass:  # only update password if a new one was provided
        current_user.smtp_app_password = smtp_pass
    elif not smtp_email:  # if email cleared, clear password too
        current_user.smtp_app_password = None

    db.session.commit()

    # Test the credentials if both provided
    if smtp_email and smtp_pass:
        import smtplib
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(smtp_email, smtp_pass)
            return jsonify({'success': True, 'message': '✅ Email settings saved and verified! Your Gmail is connected.'})
        except Exception as e:
            # Save anyway but warn
            return jsonify({'success': True, 'verified': False,
                'message': f'⚠️ Settings saved but Gmail test failed: {str(e)}. Check your App Password.'})

    return jsonify({'success': True, 'message': 'Email settings saved.'})

@app.route('/api/room-status/<room_code>')
@login_required
def room_status(room_code):
    """Check if a recruiter is present in the given room (for candidate waiting screen)."""
    sess = InterviewSession.query.filter_by(room_code=room_code).first()
    if not sess:
        return jsonify({'success': False, 'message': 'Room not found'}), 404
    # A recruiter is considered present if session status is 'active'
    recruiter_present = sess.status == 'active'
    scheduled_at = None
    if sess.created_at:
        scheduled_at = sess.created_at.strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({
        'success': True,
        'recruiter_present': recruiter_present,
        'status': sess.status,
        'scheduled_at': scheduled_at,
        'job_role': sess.job_role,
        'mode': sess.mode,
    })

@app.route('/api/my-scheduled-interviews')
@login_required
def my_scheduled_interviews():
    if current_user.is_admin:
        scheds = ScheduledInterview.query.filter_by(recruiter_id=current_user.id).order_by(ScheduledInterview.scheduled_at.desc()).all()
    else:
        scheds = ScheduledInterview.query.filter_by(candidate_id=current_user.id).order_by(ScheduledInterview.scheduled_at.desc()).all()
    return jsonify({'success': True, 'scheduled': [s.to_dict() for s in scheds]})


@app.route('/api/interview-complete/<int:session_id>')
@login_required
def interview_complete_info(session_id):
    """Returns info for the post-interview completion page."""
    sess = db.session.get(InterviewSession, session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    if sess.candidate_id != current_user.id and not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    sched = ScheduledInterview.query.filter_by(session_id=session_id).first()
    sub = TestSubmission.query.filter_by(session_id=session_id).first()

    return jsonify({
        'success': True,
        'session': sess.to_dict(),
        'scheduled': sched.to_dict() if sched else None,
        'submission': sub.to_dict() if sub else None,
    })


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)