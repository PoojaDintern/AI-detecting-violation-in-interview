from extensions import db
from datetime import datetime
import json


class TestSubmission(db.Model):
    __tablename__ = 'test_submissions'
    id                    = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id               = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id            = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='SET NULL'), nullable=True)
    job_role              = db.Column(db.String(100), default='General')
    mode                  = db.Column(db.String(30),  default='mcq')
    answers               = db.Column(db.Text, nullable=False)
    credibility_score     = db.Column(db.Integer, default=100)
    interview_score       = db.Column(db.Integer, default=0)
    total_violations      = db.Column(db.Integer, default=0)
    submitted_at          = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    exam_duration_seconds = db.Column(db.Integer)
    passed                = db.Column(db.Boolean, default=False)
    attempt_number        = db.Column(db.Integer, default=1)
    ai_feedback           = db.Column(db.Text)
    round_number          = db.Column(db.Integer, default=1)
    round_name            = db.Column(db.String(100), default='Round 1')
    posting_id            = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='SET NULL'), nullable=True)

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
            'round_number': self.round_number or 1,
            'round_name': self.round_name or 'Round 1',
            'posting_id': self.posting_id,
        }
