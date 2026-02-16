# config.py - Windows Authentication (No password needed!)
import os
import urllib.parse

class Config:
    """Application configuration"""
    
    # Flask Secret Key
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # ========================================
    # SESSION CONFIGURATION - ADD THIS SECTION
    # ========================================
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_NAME = 'proctoring_session'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour in seconds
    SESSION_TYPE = 'filesystem'  # or 'sqlalchemy' if you prefer
    
    # ========================================
    # MSSQL Database Configuration
    # WINDOWS AUTHENTICATION (No password!)
    # ========================================
    
    DB_SERVER = 'localhost\\SQLEXPRESS'  # ← Change if your server name is different
    DB_NAME = 'ProctoringDB'
    
    # Build connection string for Windows Authentication
    params = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={DB_SERVER};'
        f'DATABASE={DB_NAME};'
        'Trusted_Connection=yes;'  # ← This uses Windows Authentication
        'TrustServerCertificate=yes;'
    )
    
    SQLALCHEMY_DATABASE_URI = f"mssql+pyodbc:///?odbc_connect={params}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set to True for debugging SQL queries
    
    # ========================================
    # Proctoring Settings
    # ========================================
    
    MAX_TAB_SWITCHES = 3
    MAX_FULLSCREEN_EXITS = 2
    FACE_CHECK_INTERVAL = 3000  # milliseconds
    CREDIBILITY_PASS_THRESHOLD = 60  # minimum score to pass
    
    # Violation Severity Points
    SEVERITY_POINTS = {
        1: 5,   # Low severity
        2: 10,  # Medium severity
        3: 15   # High severity
    }
    
    # Upload folder for profile pictures (future use)
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # PDF Report Settings
    REPORTS_FOLDER = 'reports'
    
    @staticmethod
    def init_app(app):
        """Initialize application"""
        # Create necessary folders
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)