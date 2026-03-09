from extensions import db
from datetime import datetime


class InterviewerAssignment(db.Model):
    """Tracks external interviewers assigned by a recruiter to a specific interview round."""
    __tablename__ = 'interviewer_assignments'
    id                   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id           = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=False)
    scheduled_id         = db.Column(db.Integer, db.ForeignKey('scheduled_interviews.id', ondelete='CASCADE'), nullable=True)
    recruiter_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    interviewer_name     = db.Column(db.String(150), nullable=False)
    interviewer_email    = db.Column(db.String(120), nullable=False)
    interviewer_user_id  = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    round_number         = db.Column(db.Integer, default=1)
    round_name           = db.Column(db.String(100), default='Round 1')
    notes                = db.Column(db.Text)
    interviewer_score    = db.Column(db.Integer)
    interviewer_feedback = db.Column(db.Text)
    status               = db.Column(db.String(20), default='assigned')
    email_sent           = db.Column(db.Boolean, default=False)
    created_at           = db.Column(db.DateTime, default=datetime.utcnow)

    session          = db.relationship('InterviewSession', foreign_keys=[session_id])
    recruiter        = db.relationship('User', foreign_keys=[recruiter_id])
    interviewer_user = db.relationship('User', foreign_keys=[interviewer_user_id])

    def to_dict(self):
        s = self.session
        return {
            'id': self.id,
            'session_id': self.session_id,
            'scheduled_id': self.scheduled_id,
            'room_code': s.room_code if s else '',
            'job_role': s.job_role if s else '',
            'candidate_name': (s.candidate.full_name or s.candidate.username) if s else '',
            'interviewer_name': self.interviewer_name,
            'interviewer_email': self.interviewer_email,
            'recruiter_name': self.recruiter.full_name or self.recruiter.username,
            'round_number': self.round_number or 1,
            'round_name': self.round_name or 'Round 1',
            'notes': self.notes or '',
            'interviewer_score': self.interviewer_score,
            'interviewer_feedback': self.interviewer_feedback or '',
            'status': self.status,
            'email_sent': self.email_sent,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'session_status': s.status if s else '',
            'scheduled_at': s.created_at.strftime('%Y-%m-%d %H:%M:%S') if s else '',
        }
