from extensions import db
from datetime import datetime


class Violation(db.Model):
    __tablename__ = 'violations'
    id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    session_id     = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True)
    violation_type = db.Column(db.String(50), nullable=False, index=True)
    timestamp      = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    severity       = db.Column(db.Integer, default=1)
    description    = db.Column(db.Text)
    gaze_data      = db.Column(db.Text)
    device_data    = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id, 'user_id': self.user_id,
            'username': self.user.username,
            'violation_type': self.violation_type,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'severity': self.severity, 'description': self.description,
        }


class GazeEvent(db.Model):
    __tablename__ = 'gaze_events'
    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True)
    direction  = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)


class DeviceAlert(db.Model):
    __tablename__ = 'device_alerts'
    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id  = db.Column(db.Integer, db.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True)
    device_type = db.Column(db.String(50))
    confidence  = db.Column(db.Float)
    image_b64   = db.Column(db.Text)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
