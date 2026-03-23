from extensions import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id                = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username          = db.Column(db.String(80),  unique=True, nullable=False, index=True)
    email             = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash     = db.Column(db.String(512), nullable=False)
    full_name         = db.Column(db.String(150))
    role              = db.Column(db.String(20), default='candidate')
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)
    company_name      = db.Column(db.String(200))
    phone             = db.Column(db.String(30))
    company_about     = db.Column(db.Text)
    company_website   = db.Column(db.String(200))
    company_industry  = db.Column(db.String(100))
    company_size      = db.Column(db.String(50))
    logo_url          = db.Column(db.Text)
    smtp_email        = db.Column(db.String(120))
    smtp_app_password = db.Column(db.String(256))
    photo_url         = db.Column(db.Text)  # base64 or URL of candidate profile photo

    violations  = db.relationship('Violation',      backref='user', lazy='dynamic', cascade='all,delete-orphan')
    submissions = db.relationship('TestSubmission', backref='user', lazy='dynamic', cascade='all,delete-orphan')

    @property
    def is_admin(self):
        # Admin = recruiter (no separate admin role)
        return self.role == 'recruiter'

    @property
    def is_recruiter(self):
        return self.role == 'recruiter' 

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw, method='scrypt')

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)