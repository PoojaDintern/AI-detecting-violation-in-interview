from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from extensions import db
from models import (InterviewPipeline, JobRoundConfig, RoundConfigDetail, CandidateCooldown)

pipeline_bp = Blueprint('pipeline', __name__)


@pipeline_bp.route('/api/round-config/<int:posting_id>', methods=['GET'])
@login_required
def get_round_config(posting_id):
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    config = JobRoundConfig.query.filter_by(posting_id=posting_id).first()
    if not config:
        return jsonify({'success': True, 'config': None, 'default_rounds': 3})
    return jsonify({'success': True, 'config': config.to_dict()})


@pipeline_bp.route('/api/round-config/save', methods=['POST'])
@login_required
def save_round_config():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data         = request.get_json() or {}
    posting_id   = data.get('posting_id')
    total_rounds = data.get('total_rounds', 3)
    rounds_data  = data.get('rounds', [])

    if not posting_id:
        return jsonify({'success': False, 'message': 'posting_id required'}), 400
    if not (1 <= total_rounds <= 5):
        return jsonify({'success': False, 'message': 'total_rounds must be 1–5'}), 400
    if len(rounds_data) != total_rounds:
        return jsonify({'success': False, 'message': f'Expected {total_rounds} round definitions'}), 400

    try:
        config = JobRoundConfig.query.filter_by(posting_id=posting_id).first()
        if config:
            RoundConfigDetail.query.filter_by(config_id=config.id).delete()
            config.total_rounds = total_rounds
            config.updated_at   = datetime.utcnow()
        else:
            config = JobRoundConfig(posting_id=posting_id, recruiter_id=current_user.id,
                                    total_rounds=total_rounds)
            db.session.add(config)
            db.session.flush()

        for r in rounds_data:
            detail = RoundConfigDetail(
                config_id         = config.id,
                round_number      = r.get('round_number'),
                round_name        = r.get('round_name', f"Round {r.get('round_number')}"),
                interview_mode    = r.get('interview_mode', 'mcq'),
                pass_threshold    = r.get('pass_threshold', 60),
                interviewer_name  = (r.get('interviewer_name') or '').strip() or None,
                interviewer_email = (r.get('interviewer_email') or '').strip().lower() or None,
            )
            db.session.add(detail)

        db.session.commit()
        return jsonify({'success': True, 'config': config.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@pipeline_bp.route('/api/round-config/all', methods=['GET'])
@login_required
def get_all_round_configs():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    configs = JobRoundConfig.query.filter_by(recruiter_id=current_user.id).all()
    return jsonify({'success': True, 'configs': [c.to_dict() for c in configs]})


@pipeline_bp.route('/api/pipeline/<int:candidate_id>/<int:posting_id>', methods=['GET'])
@login_required
def get_pipeline(candidate_id, posting_id):
    if not current_user.is_recruiter and current_user.id != candidate_id:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    pipeline = InterviewPipeline.query.filter_by(
        candidate_id=candidate_id, posting_id=posting_id).first()
    if not pipeline:
        return jsonify({'success': True, 'pipeline': None})
    return jsonify({'success': True, 'pipeline': pipeline.to_dict()})


@pipeline_bp.route('/api/pipeline/my/<int:posting_id>', methods=['GET'])
@login_required
def get_my_pipeline(posting_id):
    try:
        pipeline = InterviewPipeline.query.filter_by(
            candidate_id=current_user.id, posting_id=posting_id).first()
        if not pipeline:
            return jsonify({'success': True, 'pipeline': None})
        return jsonify({'success': True, 'pipeline': pipeline.to_dict()})
    except Exception:
        return jsonify({'success': True, 'pipeline': None})


@pipeline_bp.route('/api/pipeline/unlock-round', methods=['POST'])
@login_required
def unlock_next_round():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data         = request.get_json() or {}
    candidate_id = data.get('candidate_id')
    posting_id   = data.get('posting_id')
    round_number = data.get('round_number')
    if not all([candidate_id, posting_id, round_number]):
        return jsonify({'success': False, 'message': 'candidate_id, posting_id, round_number required'}), 400
    pipeline = InterviewPipeline.query.filter_by(
        candidate_id=candidate_id, posting_id=posting_id).first()
    if not pipeline:
        return jsonify({'success': False, 'message': 'Pipeline not found'}), 404
    rounds = pipeline.get_rounds()
    rkey = str(round_number)
    if rkey not in rounds:
        rounds[rkey] = {}
    rounds[rkey]['status'] = 'pending'
    pipeline.set_rounds(rounds)
    pipeline.current_round = round_number
    pipeline.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'message': f'Round {round_number} unlocked'})


@pipeline_bp.route('/api/pipeline/reset-round', methods=['POST'])
@login_required
def reset_round():
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    data         = request.get_json() or {}
    candidate_id = data.get('candidate_id')
    posting_id   = data.get('posting_id')
    round_number = data.get('round_number')
    pipeline = InterviewPipeline.query.filter_by(
        candidate_id=candidate_id, posting_id=posting_id).first()
    if not pipeline:
        return jsonify({'success': False, 'message': 'Pipeline not found'}), 404
    rounds = pipeline.get_rounds()
    rkey = str(round_number)
    rounds[rkey] = {'status': 'pending', 'session_id': None, 'submission_id': None, 'score': None}
    pipeline.set_rounds(rounds)
    pipeline.overall_status = 'in_progress'
    pipeline.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'message': f'Round {round_number} reset'})


@pipeline_bp.route('/api/candidate/cooldown-status', methods=['GET'])
@login_required
def candidate_cooldown_status():
    cooldown = CandidateCooldown.query.filter_by(
        candidate_id=current_user.id, is_active=True).first()
    if not cooldown:
        return jsonify({'success': True, 'on_cooldown': False})
    if datetime.utcnow() >= cooldown.eligible_at:
        cooldown.is_active = False
        db.session.commit()
        return jsonify({'success': True, 'on_cooldown': False})
    return jsonify({'success': True, 'on_cooldown': True, 'cooldown': cooldown.to_dict()})


@pipeline_bp.route('/api/pipeline/all-for-posting/<int:posting_id>', methods=['GET'])
@login_required
def get_all_pipelines_for_posting(posting_id):
    if not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    pipelines = InterviewPipeline.query.filter_by(posting_id=posting_id).all()
    return jsonify({'success': True, 'pipelines': [p.to_dict() for p in pipelines]})