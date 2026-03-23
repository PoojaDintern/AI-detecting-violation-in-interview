from extensions import db
from datetime import datetime


class TokenUsage(db.Model):
    __tablename__ = 'token_usage'

    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # Who triggered it
    recruiter_id  = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    session_id    = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='SET NULL'), nullable=True)
    # What kind of call
    call_type     = db.Column(db.String(30),  nullable=False)
    # call_type values: 'questions' | 'evaluations' | 'final' | 'mcq_questions'
    label         = db.Column(db.String(150))  # job role or context
    # Token counts
    prompt_tokens    = db.Column(db.Integer, default=0)   # input tokens
    response_tokens  = db.Column(db.Integer, default=0)   # output tokens
    total_tokens     = db.Column(db.Integer, default=0)   # total
    # Prompt stored for reference
    prompt_preview   = db.Column(db.String(300))  # first 300 chars of prompt
    # Timestamp
    called_at     = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id':              self.id,
            'call_type':       self.call_type,
            'label':           self.label or '',
            'prompt_tokens':   self.prompt_tokens,
            'response_tokens': self.response_tokens,
            'total_tokens':    self.total_tokens,
            'prompt_preview':  self.prompt_preview or '',
            'called_at':       self.called_at.strftime('%Y-%m-%d %H:%M:%S') if self.called_at else '',
        }