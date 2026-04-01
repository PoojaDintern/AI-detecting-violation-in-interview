import os
from dotenv import load_dotenv

# Load .env file (only in development — in production set env vars directly)
load_dotenv()

# ── Flask ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set. Add it to your .env file.")

SESSION_COOKIE_HTTPONLY  = True
SESSION_COOKIE_SAMESITE  = 'Lax'
SESSION_COOKIE_SECURE    = False  # Must be False for localhost HTTP development
SESSION_PERMANENT = True
PERMANENT_SESSION_LIFETIME = 86400  # 24 hours

# ── Supabase PostgreSQL ────────────────────────────────────────────────────────
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
if not SQLALCHEMY_DATABASE_URI:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file.")

SQLALCHEMY_ECHO                = False
SQLALCHEMY_TRACK_MODIFICATIONS = False

# ── Gemini AI ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
AI_MODEL       = 'gemini-1.5-flash'

# ── App Base URL (for email links, interviewer portal) ────────────────────────
BASE_URL = os.environ.get('BASE_URL', 'http://localhost:5000')

# ── Scoring ───────────────────────────────────────────────────────────────────
CREDIBILITY_PASS_THRESHOLD  = 60
SEVERITY_POINTS             = {1: 5, 2: 10, 3: 15}
EXCESS_VIOLATION_PENALTY    = 2
EXCESS_VIOLATION_THRESHOLD  = 10

# ── Proctoring thresholds ─────────────────────────────────────────────────────
FACE_CHECK_INTERVAL         = 3000
GAZE_CHECK_INTERVAL         = 4000
DEVICE_CHECK_INTERVAL       = 6000
GAZE_AWAY_THRESHOLD         = 0.35
PHONE_DETECTION_CONFIDENCE  = 0.70
PHONE_MIN_CONTOUR_AREA      = 3000
PHONE_MAX_FRAME_RATIO       = 0.40
PHONE_ASPECT_RATIO_MIN      = 1.4
PHONE_ASPECT_RATIO_MAX      = 2.8

# ── Job Roles ─────────────────────────────────────────────────────────────────
JOB_ROLES = [
    'Python Developer', 'Frontend Developer', 'Backend Developer',
    'Full Stack Developer', 'Data Scientist', 'Machine Learning Engineer',
    'DevOps Engineer', 'Mobile Developer', 'QA Engineer', 'Product Manager',
    'UI/UX Designer', 'Cybersecurity Analyst', 'Cloud Architect',
    'Database Administrator', 'Business Analyst',
]

# ── WebRTC ICE Servers ────────────────────────────────────────────────────────
WEBRTC_ICE_SERVERS = [
    {'urls': 'stun:stun.l.google.com:19302'},
    {'urls': 'stun:stun1.l.google.com:19302'},
    {'urls': 'stun:stun2.l.google.com:19302'},
    {'urls': 'stun:stun3.l.google.com:19302'},
    {'urls': 'stun:stun4.l.google.com:19302'},
    {'urls': 'turn:a.relay.metered.ca:80',             'username': 'openrelayproject', 'credential': 'openrelayproject'},
    {'urls': 'turn:a.relay.metered.ca:80?transport=tcp','username': 'openrelayproject', 'credential': 'openrelayproject'},
    {'urls': 'turn:a.relay.metered.ca:443',            'username': 'openrelayproject', 'credential': 'openrelayproject'},
    {'urls': 'turn:a.relay.metered.ca:443?transport=tcp','username': 'openrelayproject','credential': 'openrelayproject'},
    {'urls': 'turn:openrelay.metered.ca:80',           'username': 'openrelayproject', 'credential': 'openrelayproject'},
    {'urls': 'turn:openrelay.metered.ca:443',          'username': 'openrelayproject', 'credential': 'openrelayproject'},
    {'urls': 'turn:openrelay.metered.ca:443?transport=tcp','username': 'openrelayproject','credential': 'openrelayproject'},
]


MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

class Config:
    SECRET_KEY                     = SECRET_KEY
    SESSION_COOKIE_HTTPONLY        = SESSION_COOKIE_HTTPONLY
    SESSION_COOKIE_SAMESITE        = SESSION_COOKIE_SAMESITE
    SESSION_COOKIE_SECURE          = SESSION_COOKIE_SECURE
    PERMANENT_SESSION_LIFETIME     = PERMANENT_SESSION_LIFETIME
    MAX_CONTENT_LENGTH             = MAX_CONTENT_LENGTH
    SQLALCHEMY_DATABASE_URI        = SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_ECHO                = SQLALCHEMY_ECHO
    SQLALCHEMY_TRACK_MODIFICATIONS = SQLALCHEMY_TRACK_MODIFICATIONS
    GEMINI_API_KEY                 = GEMINI_API_KEY
    PLUGIN_API_KEYS                = [os.environ.get('PLUGIN_API_KEY', '2f1ea8d9d0ce7e64eb6ac5a5890e16e2583a11db5a1e2e5e57dfa18c57f681cb')]
    AI_MODEL                       = AI_MODEL
    BASE_URL                       = BASE_URL
    CREDIBILITY_PASS_THRESHOLD     = CREDIBILITY_PASS_THRESHOLD
    SEVERITY_POINTS                = SEVERITY_POINTS
    EXCESS_VIOLATION_PENALTY       = EXCESS_VIOLATION_PENALTY
    EXCESS_VIOLATION_THRESHOLD     = EXCESS_VIOLATION_THRESHOLD
    FACE_CHECK_INTERVAL            = FACE_CHECK_INTERVAL
    GAZE_CHECK_INTERVAL            = GAZE_CHECK_INTERVAL
    DEVICE_CHECK_INTERVAL          = DEVICE_CHECK_INTERVAL
    GAZE_AWAY_THRESHOLD            = GAZE_AWAY_THRESHOLD
    PHONE_DETECTION_CONFIDENCE     = PHONE_DETECTION_CONFIDENCE
    PHONE_MIN_CONTOUR_AREA         = PHONE_MIN_CONTOUR_AREA
    PHONE_MAX_FRAME_RATIO          = PHONE_MAX_FRAME_RATIO
    PHONE_ASPECT_RATIO_MIN         = PHONE_ASPECT_RATIO_MIN
    PHONE_ASPECT_RATIO_MAX         = PHONE_ASPECT_RATIO_MAX
    JOB_ROLES                      = JOB_ROLES
    WEBRTC_ICE_SERVERS             = WEBRTC_ICE_SERVERS

    @staticmethod
    def init_app(app):
        pass