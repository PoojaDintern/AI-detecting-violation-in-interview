"""
RecruitAI — Main Application Entry Point
Modular structure:
  app.py              → app factory + startup
  config.py           → all configuration
  extensions.py       → shared Flask extensions (db, socketio, etc.)
  models/             → one file per database model group
  routes/             → one Blueprint file per feature area
  utils/              → helpers, AI, email, proctoring utilities
  seed.py             → question bank seeding
  templates/          → all HTML files
"""
import os
import traceback
from flask import Flask, request, jsonify, send_from_directory, redirect
from config import Config
from extensions import db, login_manager, socketio, cors

# ── Import all models so SQLAlchemy picks them up ──────────────────────────────
import models  # noqa: F401 — registers all ORM classes

# ── Import all route blueprints ────────────────────────────────────────────────
from routes import (auth_bp, pages_bp, sessions_bp, proctoring_bp, ai_bp,
                    jobs_bp, results_bp, pipeline_bp, interviewer_bp,
                    recruiter_bp, register_socketio_events)


def create_app():
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    app = Flask(__name__, static_folder=templates_dir, template_folder=templates_dir)
    app.config.from_object(Config)
    Config.init_app(app)

    # ── CORS ──────────────────────────────────────────────────────────────────
    _allowed_origins = ["http://localhost:5000", "http://127.0.0.1:5000"]
    _base_url = os.environ.get("BASE_URL", "")
    if _base_url and _base_url not in _allowed_origins:
        _allowed_origins.append(_base_url)
    cors.init_app(app, supports_credentials=True, origins=_allowed_origins)

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins='*', async_mode='eventlet',
                      allow_upgrades=True, ping_timeout=120, ping_interval=25)

    # ── Login Manager ─────────────────────────────────────────────────────────
    login_manager.init_app(app)
    login_manager.login_view = 'pages.serve_index'
    login_manager.session_protection = 'strong'

    from models.user import User

    @login_manager.user_loader
    def load_user(uid):
        return db.session.get(User, int(uid))

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': 'Login required', 'code': 'not_authenticated'}), 401
        return redirect('/')

    # ── Error Handlers ────────────────────────────────────────────────────────
    @app.errorhandler(500)
    def handle_500(e):
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

    @app.errorhandler(404)
    def handle_404(e):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': 'Endpoint not found'}), 404
        return redirect('/')

    @app.errorhandler(401)
    def handle_401(e):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401

    @app.errorhandler(403)
    def handle_403(e):
        return jsonify({'success': False, 'message': 'Forbidden'}), 403

    # ── Register Blueprints ───────────────────────────────────────────────────
    for bp in [auth_bp, pages_bp, sessions_bp, proctoring_bp, ai_bp,
               jobs_bp, results_bp, pipeline_bp, interviewer_bp, recruiter_bp]:
        app.register_blueprint(bp)

    # ── Register SocketIO Events ──────────────────────────────────────────────
    register_socketio_events(app)

    return app


def init_database(app):
    """Initialize DB tables and seed default data."""
    from models.user import User
    from seed import seed_questions
    from sqlalchemy import text

    with app.app_context():
        db.create_all()

        if not User.query.filter_by(username='admin').first():
            u = User(username='admin', email='admin@proctoring.com',
                     full_name='Administrator', role='admin')
            u.set_password('admin123')
            db.session.add(u)
            db.session.commit()

        if not User.query.filter_by(username='recruiter').first():
            u = User(username='recruiter', email='recruiter@proctoring.com',
                     full_name='Demo Recruiter', role='recruiter')
            u.set_password('recruiter123')
            db.session.add(u)
            db.session.commit()

        if not User.query.filter_by(username='student').first():
            u = User(username='student', email='student@test.com',
                     full_name='Test Student', role='candidate')
            u.set_password('student123')
            db.session.add(u)
            db.session.commit()

        seed_questions()

        # Safe column migration (add mcq_source if missing)
        try:
            with db.engine.connect() as conn:
                conn.execute(text(
                    "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS mcq_source VARCHAR(20) DEFAULT 'hardcoded'"
                ))
                conn.commit()
        except Exception:
            pass

        print("✅ Database ready")


# ── Application instance ───────────────────────────────────────────────────────
app = create_app()
init_database(app)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
