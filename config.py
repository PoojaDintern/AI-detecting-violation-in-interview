import os

# Flask
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
PERMANENT_SESSION_LIFETIME = 7200  # 2 hours

# ── Supabase PostgreSQL Database ───────────────────────────────────────────────
# Replace YOUR-PASSWORD with your actual Supabase database password
SQLALCHEMY_DATABASE_URI = os.environ.get(
    'DATABASE_URL',
    'postgresql://postgres:poojad01252006d@db.vlvpnxbctzxlljqapqyw.supabase.co:5432/postgres'
)
SQLALCHEMY_ECHO = False
SQLALCHEMY_TRACK_MODIFICATIONS = False

# ── Gemini AI ──────────────────────────────────────────────────────────────────
# Get your key from: https://aistudio.google.com/app/apikey
# Set before running: set GEMINI_API_KEY=AIza-your-key-here
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
AI_MODEL = 'gemini-1.5-flash'

# Scoring
CREDIBILITY_PASS_THRESHOLD = 60
SEVERITY_POINTS = {1: 5, 2: 10, 3: 15}
EXCESS_VIOLATION_PENALTY = 2
EXCESS_VIOLATION_THRESHOLD = 10

# Proctoring thresholds
FACE_CHECK_INTERVAL = 3000
GAZE_CHECK_INTERVAL = 4000
DEVICE_CHECK_INTERVAL = 6000
GAZE_AWAY_THRESHOLD = 0.35
PHONE_DETECTION_CONFIDENCE = 0.70
PHONE_MIN_CONTOUR_AREA = 3000
PHONE_MAX_FRAME_RATIO = 0.40
PHONE_ASPECT_RATIO_MIN = 1.4
PHONE_ASPECT_RATIO_MAX = 2.8

# Job Roles
JOB_ROLES = [
    'Python Developer',
    'Frontend Developer',
    'Backend Developer',
    'Full Stack Developer',
    'Data Scientist',
    'Machine Learning Engineer',
    'DevOps Engineer',
    'Mobile Developer',
    'QA Engineer',
    'Product Manager',
    'UI/UX Designer',
    'Cybersecurity Analyst',
    'Cloud Architect',
    'Database Administrator',
    'Business Analyst',
]

# WebRTC ICE Servers
WEBRTC_ICE_SERVERS = [
    # STUN servers - helps discover public IP
    {'urls': 'stun:stun.l.google.com:19302'},
    {'urls': 'stun:stun1.l.google.com:19302'},
    {'urls': 'stun:stun2.l.google.com:19302'},
    {'urls': 'stun:stun3.l.google.com:19302'},
    # Free TURN servers - relays traffic when direct connection fails (cross-network)
    {
        'urls': 'turn:openrelay.metered.ca:80',
        'username': 'openrelayproject',
        'credential': 'openrelayproject'
    },
    {
        'urls': 'turn:openrelay.metered.ca:443',
        'username': 'openrelayproject',
        'credential': 'openrelayproject'
    },
    {
        'urls': 'turn:openrelay.metered.ca:443?transport=tcp',
        'username': 'openrelayproject',
        'credential': 'openrelayproject'
    },
]


class Config:
    SECRET_KEY                     = SECRET_KEY
    SESSION_COOKIE_HTTPONLY        = SESSION_COOKIE_HTTPONLY
    SESSION_COOKIE_SAMESITE        = SESSION_COOKIE_SAMESITE
    SESSION_COOKIE_SECURE          = SESSION_COOKIE_SECURE
    PERMANENT_SESSION_LIFETIME     = PERMANENT_SESSION_LIFETIME
    SQLALCHEMY_DATABASE_URI        = SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_ECHO                = SQLALCHEMY_ECHO
    SQLALCHEMY_TRACK_MODIFICATIONS = SQLALCHEMY_TRACK_MODIFICATIONS
    GEMINI_API_KEY                 = GEMINI_API_KEY
    AI_MODEL                       = AI_MODEL
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