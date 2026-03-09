from .auth import auth_bp
from .pages import pages_bp
from .sessions import sessions_bp
from .proctoring import proctoring_bp
from .ai_routes import ai_bp
from .jobs import jobs_bp
from .results import results_bp
from .pipeline import pipeline_bp
from .interviewer import interviewer_bp
from .recruiter import recruiter_bp
from .socketio_events import register_socketio_events

__all__ = [
    'auth_bp', 'pages_bp', 'sessions_bp', 'proctoring_bp',
    'ai_bp', 'jobs_bp', 'results_bp', 'pipeline_bp',
    'interviewer_bp', 'recruiter_bp', 'register_socketio_events',
]
