from extensions import db
from datetime import datetime
import json


class InterviewSession(db.Model):
    __tablename__ = 'interview_sessions'
    id                = db.Column(db.Integer, primary_key=True, autoincrement=True)
    candidate_id      = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    recruiter_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    job_role          = db.Column(db.String(100), nullable=False)
    mode              = db.Column(db.String(30),  nullable=False)
    room_code         = db.Column(db.String(16),  unique=True, nullable=False)
    status            = db.Column(db.String(20),  default='pending')
    credibility_score = db.Column(db.Integer, default=100)
    interview_score   = db.Column(db.Integer, default=0)
    started_at        = db.Column(db.DateTime)
    ended_at          = db.Column(db.DateTime)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)
    question_ids      = db.Column(db.Text)
    ai_transcript     = db.Column(db.Text)
    recruiter_notes   = db.Column(db.Text)
    round_number      = db.Column(db.Integer, default=1)
    round_name        = db.Column(db.String(100))
    posting_id        = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='SET NULL'), nullable=True)
    parent_session_id = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='SET NULL'), nullable=True)
    mcq_source        = db.Column(db.String(20), default='hardcoded')

    candidate = db.relationship('User', foreign_keys=[candidate_id], backref='sessions_as_candidate')
    recruiter = db.relationship('User', foreign_keys=[recruiter_id], backref='sessions_as_recruiter')

    def to_dict(self):
        try:
            cname = (self.candidate.full_name or self.candidate.username) if self.candidate else 'Candidate'
            cusername = self.candidate.username if self.candidate else ''
        except Exception:
            cname = 'Candidate'
            cusername = ''
        try:
            created_at_str = self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        except Exception:
            created_at_str = ''
        return {
            'id': self.id,
            'candidate_name': cname,
            'candidate_username': cusername,
            'job_role': self.job_role or '',
            'mode': self.mode or 'mcq',
            'room_code': self.room_code or '',
            'status': self.status or 'pending',
            'credibility_score': self.credibility_score or 100,
            'interview_score': self.interview_score or 0,
            'started_at': self.started_at.strftime('%Y-%m-%d %H:%M:%S') if self.started_at else None,
            'ended_at':   self.ended_at.strftime('%Y-%m-%d %H:%M:%S') if self.ended_at else None,
            'created_at': created_at_str,
            'round_number': self.round_number or 1,
            'round_name': self.round_name or 'Round 1',
            'posting_id': self.posting_id,
        }


class InterviewPipeline(db.Model):
    """Tracks a candidate's full round-by-round progress for a specific job posting."""
    __tablename__ = 'interview_pipeline'
    id                   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    candidate_id         = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    posting_id           = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='CASCADE'), nullable=False)
    application_id       = db.Column(db.Integer, db.ForeignKey('job_applications.id', ondelete='CASCADE'), nullable=True)
    config_id            = db.Column(db.Integer, db.ForeignKey('job_round_config.id', ondelete='SET NULL'), nullable=True)
    total_rounds         = db.Column(db.Integer, default=3)
    current_round        = db.Column(db.Integer, default=1)
    overall_status       = db.Column(db.String(30), default='in_progress')
    failed_all_rounds_at = db.Column(db.DateTime, nullable=True)
    eligible_again_at    = db.Column(db.DateTime, nullable=True)
    created_at           = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at           = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    rounds_data          = db.Column(db.Text, default='{}')

    candidate = db.relationship('User', foreign_keys=[candidate_id])
    __table_args__ = (db.UniqueConstraint('candidate_id', 'posting_id', name='uq_pipeline_candidate_posting'),)

    def get_rounds(self):
        try:
            return json.loads(self.rounds_data) if self.rounds_data else {}
        except Exception:
            return {}

    def set_rounds(self, d):
        self.rounds_data = json.dumps(d)

    def to_dict(self):
        rounds = self.get_rounds()
        return {
            'id': self.id,
            'candidate_id': self.candidate_id,
            'posting_id': self.posting_id,
            'application_id': self.application_id,
            'total_rounds': self.total_rounds,
            'current_round': self.current_round,
            'overall_status': self.overall_status,
            'rounds': rounds,
            'failed_all_rounds_at': self.failed_all_rounds_at.strftime('%Y-%m-%d %H:%M:%S') if self.failed_all_rounds_at else None,
            'eligible_again_at': self.eligible_again_at.strftime('%Y-%m-%d %H:%M:%S') if self.eligible_again_at else None,
        }


class CandidateCooldown(db.Model):
    """Global 60-day ban after a candidate fails all rounds for any posting."""
    __tablename__ = 'candidate_cooldowns'
    id                      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    candidate_id            = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True)
    triggered_by_posting_id = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='SET NULL'), nullable=True)
    triggered_at            = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    eligible_at             = db.Column(db.DateTime, nullable=False)
    is_active               = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'candidate_id': self.candidate_id,
            'triggered_at': self.triggered_at.strftime('%Y-%m-%d %H:%M:%S'),
            'eligible_at': self.eligible_at.strftime('%Y-%m-%d %H:%M:%S'),
            'is_active': self.is_active,
            'days_remaining': max(0, (self.eligible_at - datetime.utcnow()).days),
        }
