from extensions import db
from datetime import datetime
import json


class JobPosting(db.Model):
    __tablename__ = 'job_postings'
    id                  = db.Column(db.Integer, primary_key=True, autoincrement=True)
    recruiter_id        = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    job_section         = db.Column(db.String(100), nullable=False)
    job_role            = db.Column(db.String(100), nullable=False)
    job_title           = db.Column(db.String(200))
    company_name        = db.Column(db.String(150), nullable=False)
    description         = db.Column(db.Text)
    skills_required     = db.Column(db.Text)
    experience_required = db.Column(db.String(50))
    job_type            = db.Column(db.String(50))
    salary_package      = db.Column(db.String(100))
    work_mode           = db.Column(db.String(30))
    is_active           = db.Column(db.Boolean, default=True)
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)

    recruiter    = db.relationship('User', foreign_keys=[recruiter_id])
    applications = db.relationship('JobApplication', backref='posting', lazy='dynamic', cascade='all,delete-orphan')

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
            'salary_package': self.salary_package or '',
            'work_mode': self.work_mode or '',
            'is_active': self.is_active,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'application_count': self.applications.count(),
        }


class JobApplication(db.Model):
    __tablename__ = 'job_applications'
    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    posting_id   = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='CASCADE'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    cover_note   = db.Column(db.Text)
    status       = db.Column(db.String(30), default='applied')
    applied_at   = db.Column(db.DateTime, default=datetime.utcnow)

    candidate = db.relationship('User', foreign_keys=[candidate_id])

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
    id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    application_id = db.Column(db.Integer, db.ForeignKey('job_applications.id', ondelete='CASCADE'), nullable=False)
    session_id     = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='SET NULL'), nullable=True)
    recruiter_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    candidate_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    scheduled_at   = db.Column(db.DateTime, nullable=False)
    interview_mode = db.Column(db.String(30), default='mcq')
    calendar_link  = db.Column(db.Text)
    email_sent     = db.Column(db.Boolean, default=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    room_code      = db.Column(db.String(16))
    round_number   = db.Column(db.Integer, default=1)
    round_name     = db.Column(db.String(100), default='Round 1')

    application = db.relationship('JobApplication', foreign_keys=[application_id])
    recruiter   = db.relationship('User', foreign_keys=[recruiter_id])
    candidate   = db.relationship('User', foreign_keys=[candidate_id])
    session     = db.relationship('InterviewSession', foreign_keys=[session_id])

    def to_dict(self):
        app = self.application
        sess = self.session
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
            'room_code': self.room_code or (sess.room_code if sess else ''),
            'session_id': self.session_id,
            'session_status': sess.status if sess else 'pending',
            'session_score': sess.interview_score if sess else None,
            'session_credibility': sess.credibility_score if sess else None,
            'round_number': self.round_number or 1,
            'round_name': self.round_name or 'Round 1',
        }
