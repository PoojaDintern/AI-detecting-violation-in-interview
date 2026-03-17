from extensions import db
from datetime import datetime


class JobRoundConfig(db.Model):
    __tablename__ = 'job_round_config'
    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    posting_id   = db.Column(db.Integer, db.ForeignKey('job_postings.id', ondelete='CASCADE'), nullable=False, unique=True)
    recruiter_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    total_rounds = db.Column(db.Integer, default=3)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    details      = db.relationship('RoundConfigDetail', backref='config', cascade='all,delete-orphan', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'posting_id': self.posting_id,
            'total_rounds': self.total_rounds,
            'rounds': [d.to_dict() for d in self.details.order_by(RoundConfigDetail.round_number)],
        }


class RoundConfigDetail(db.Model):
    __tablename__ = 'round_config_details'
    id                = db.Column(db.Integer, primary_key=True, autoincrement=True)
    config_id         = db.Column(db.Integer, db.ForeignKey('job_round_config.id', ondelete='CASCADE'), nullable=False)
    round_number      = db.Column(db.Integer, nullable=False)
    round_name        = db.Column(db.String(100), nullable=False)
    interview_mode    = db.Column(db.String(30), nullable=False, default='mcq')
    pass_threshold    = db.Column(db.Integer, default=60)
    interviewer_name  = db.Column(db.String(150))
    interviewer_email = db.Column(db.String(120))

    def to_dict(self):
        return {
            'round_number':      self.round_number,
            'round_name':        self.round_name,
            'interview_mode':    self.interview_mode,
            'pass_threshold':    self.pass_threshold,
            'interviewer_name':  self.interviewer_name or '',
            'interviewer_email': self.interviewer_email or '',
        }