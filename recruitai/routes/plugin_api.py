"""
RecruitAI Interview Plugin API
================================
Exposes MCQ + AI interview engine as a REST API for external apps.
Authentication: API key via header  →  X-API-Key: <your_key>

Set PLUGIN_API_KEY in your .env file.

Endpoints:
  POST /plugin/generate-mcq       → Generate MCQ questions
  POST /plugin/score-mcq          → Score submitted MCQ answers
  POST /plugin/ai-question        → Get next AI interview question
  POST /plugin/ai-evaluate        → Evaluate a single AI answer
  POST /plugin/ai-final           → Final score + full feedback
  GET  /plugin/health             → Check plugin is alive
"""

import os
import secrets
from functools import wraps
from flask import Blueprint, request, jsonify, current_app
from models import ExamQuestion
from utils.ai import gemini_ask
from utils.helpers import parse_json_response

plugin_bp = Blueprint('plugin', __name__, url_prefix='/plugin')


# ── API Key Authentication ─────────────────────────────────────────

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key      = request.headers.get('X-API-Key') or request.args.get('api_key')
        expected = current_app.config.get('PLUGIN_API_KEY') or os.environ.get('PLUGIN_API_KEY')

        if not expected:
            return jsonify({'success': False,
                            'error': 'Plugin API key not configured on server.'}), 500
        if not key:
            return jsonify({'success': False,
                            'error': 'Missing API key. Add header: X-API-Key: <your_key>'}), 401
        if not secrets.compare_digest(key.strip(), expected.strip()):
            return jsonify({'success': False, 'error': 'Invalid API key.'}), 403

        return f(*args, **kwargs)
    return decorated


# ── Health Check ───────────────────────────────────────────────────

@plugin_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'success': True,
        'service': 'RecruitAI Interview Plugin',
        'version': '1.0',
        'endpoints': [
            'POST /plugin/generate-mcq',
            'POST /plugin/score-mcq',
            'POST /plugin/ai-question',
            'POST /plugin/ai-evaluate',
            'POST /plugin/ai-final',
        ]
    })


# ── 1. Generate MCQ Questions ──────────────────────────────────────

@plugin_bp.route('/generate-mcq', methods=['POST'])
@require_api_key
def generate_mcq():
    data       = request.get_json() or {}
    job_role   = data.get('job_role', 'Software Developer').strip()
    count      = min(int(data.get('count', 10)), 20)
    difficulty = data.get('difficulty', 'mixed')

    diff_instruction = (
        f'All questions must be {difficulty} difficulty.'
        if difficulty != 'mixed'
        else f'Mix: {count//3} easy, {count//3} medium, {count - 2*(count//3)} hard.'
    )

    prompt = f"""You are an expert technical recruiter creating a multiple-choice test for a {job_role} position.
Generate exactly {count} MCQ questions. {diff_instruction}

Return ONLY a valid JSON array (no markdown, no extra text):
[
  {{
    "id": 1,
    "question": "Question text here?",
    "option_a": "First option",
    "option_b": "Second option",
    "option_c": "Third option",
    "option_d": "Fourth option",
    "correct_answer": "A",
    "difficulty": "easy",
    "category": "relevant topic"
  }}
]
Rules: correct_answer must be A/B/C/D. Questions must be specific to {job_role}."""

    raw       = gemini_ask(current_app._get_current_object(), prompt)
    questions = parse_json_response(raw)

    if questions and isinstance(questions, list) and len(questions) > 0:
        clean = []
        for i, q in enumerate(questions):
            clean.append({
                'id':             q.get('id', i + 1),
                'question':       q.get('question', ''),
                'option_a':       q.get('option_a', ''),
                'option_b':       q.get('option_b', ''),
                'option_c':       q.get('option_c', ''),
                'option_d':       q.get('option_d', ''),
                'correct_answer': q.get('correct_answer', 'A').upper(),
                'difficulty':     q.get('difficulty', 'medium'),
                'category':       q.get('category', job_role),
            })
        return jsonify({'success': True, 'source': 'gemini',
                        'count': len(clean), 'questions': clean})

    # DB fallback
    qs = ExamQuestion.query.filter_by(job_role=job_role).limit(count).all()
    if not qs:
        qs = ExamQuestion.query.limit(count).all()
    if not qs:
        return jsonify({'success': False,
                        'error': f'No questions available for: {job_role}'}), 404

    return jsonify({'success': True, 'source': 'db_fallback',
                    'count': len(qs), 'questions': [q.to_dict() for q in qs]})


# ── 2. Score MCQ Answers ───────────────────────────────────────────

@plugin_bp.route('/score-mcq', methods=['POST'])
@require_api_key
def score_mcq():
    data           = request.get_json() or {}
    questions      = data.get('questions', [])
    answers        = data.get('answers', {})
    pass_threshold = int(data.get('pass_threshold', 60))

    if not questions:
        return jsonify({'success': False, 'error': 'No questions provided.'}), 400

    correct_count = 0
    breakdown     = []

    for q in questions:
        qid         = str(q.get('id', ''))
        correct_ans = str(q.get('correct_answer', '')).upper()
        user_ans    = str(answers.get(qid, '')).upper()
        is_correct  = (user_ans == correct_ans) and bool(user_ans)
        if is_correct:
            correct_count += 1
        breakdown.append({
            'id':             q.get('id'),
            'your_answer':    user_ans or None,
            'correct_answer': correct_ans,
            'is_correct':     is_correct,
        })

    total = len(questions)
    score = int((correct_count / max(total, 1)) * 100)

    return jsonify({
        'success':        True,
        'total':          total,
        'correct':        correct_count,
        'wrong':          total - correct_count,
        'score':          score,
        'passed':         score >= pass_threshold,
        'pass_threshold': pass_threshold,
        'breakdown':      breakdown,
    })


# ── 3. Get Next AI Interview Question ─────────────────────────────

@plugin_bp.route('/ai-question', methods=['POST'])
@require_api_key
def ai_question():
    data               = request.get_json() or {}
    job_role           = data.get('job_role', 'Software Developer').strip()
    question_number    = int(data.get('question_number', 1))
    total_questions    = int(data.get('total_questions', 5))
    previous_questions = data.get('previous_questions', [])

    prev_text = ''
    if previous_questions:
        prev_text = '\nDo NOT repeat these already-asked questions:\n' + \
                    '\n'.join(f'- {q}' for q in previous_questions)

    if question_number == 1:
        q_type      = 'introductory'
        instruction = 'Start with a warm-up or background question.'
    elif question_number == total_questions:
        q_type      = 'situational'
        instruction = 'End with a situational or behavioural question.'
    else:
        q_type      = 'technical'
        instruction = 'Ask a core technical question relevant to the role.'

    prompt = f"""You are interviewing a candidate for a {job_role} position.
This is question {question_number} of {total_questions}. {instruction}
{prev_text}

Return ONLY this JSON (no markdown, no extra text):
{{
  "question": "Your interview question here?",
  "type": "{q_type}",
  "expected_keywords": ["keyword1", "keyword2", "keyword3"],
  "follow_up": "A follow-up question, or null"
}}"""

    raw    = gemini_ask(current_app._get_current_object(), prompt)
    result = parse_json_response(raw)

    if result and 'question' in result:
        return jsonify({
            'success':           True,
            'question_number':   question_number,
            'question':          result.get('question', ''),
            'type':              result.get('type', q_type),
            'expected_keywords': result.get('expected_keywords', []),
            'follow_up':         result.get('follow_up'),
        })

    fallbacks = {
        1: f"Tell me about your experience with {job_role}.",
        2: f"What core concepts do you use daily as a {job_role}?",
        3: "Describe a challenging problem you solved recently.",
        4: "How do you ensure quality in your work?",
        5: "Where do you see yourself growing in this role?",
    }
    return jsonify({
        'success':           True,
        'question_number':   question_number,
        'question':          fallbacks.get(question_number, f"What do you know about {job_role}?"),
        'type':              q_type,
        'expected_keywords': [],
        'follow_up':         None,
    })


# ── 4. Evaluate a Single AI Answer ────────────────────────────────

@plugin_bp.route('/ai-evaluate', methods=['POST'])
@require_api_key
def ai_evaluate():
    data     = request.get_json() or {}
    job_role = data.get('job_role', 'Developer').strip()
    question = data.get('question', '').strip()
    answer   = data.get('answer', '').strip()
    keywords = data.get('expected_keywords', [])

    if not question or not answer:
        return jsonify({'success': False,
                        'error': 'Both question and answer are required.'}), 400

    prompt = f"""You are evaluating a {job_role} interview answer.

Question: {question}
Candidate's Answer: {answer}
Expected Keywords/Concepts: {', '.join(keywords) if keywords else 'N/A'}

Return ONLY this JSON (no markdown, no extra text):
{{
  "score": <integer 1-10>,
  "feedback": "<exactly 2 sentences of constructive feedback>",
  "follow_up": "<one follow-up question if incomplete, or null>",
  "strong_points": ["<what they got right>"],
  "missing_points": ["<what was missing>"]
}}"""

    raw    = gemini_ask(current_app._get_current_object(), prompt)
    result = parse_json_response(raw)

    if result and 'score' in result:
        return jsonify({'success': True, **result})

    matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
    score   = min(10, max(1, 4 + round((matches / max(len(keywords), 1)) * 6)))
    return jsonify({
        'success':        True,
        'score':          score,
        'feedback':       f'Answer recorded. {matches}/{len(keywords)} key concepts mentioned.',
        'follow_up':      None,
        'strong_points':  [],
        'missing_points': [],
    })


# ── 5. Final AI Interview Evaluation ──────────────────────────────

@plugin_bp.route('/ai-final', methods=['POST'])
@require_api_key
def ai_final():
    data           = request.get_json() or {}
    job_role       = data.get('job_role', 'Developer').strip()
    transcript     = data.get('transcript', [])
    pass_threshold = int(data.get('pass_threshold', 60))

    if not transcript:
        return jsonify({
            'success': True, 'overall_score': 0, 'technical_score': 0,
            'communication_score': 0, 'passed': False,
            'pass_threshold': pass_threshold, 'recommendation': 'No Hire',
            'summary': 'No answers were recorded.',
            'strengths': [], 'improvements': ['Complete the interview first.'],
        })

    qa_text = '\n\n'.join([
        f"Q{i+1}: {t.get('question','')}\nAnswer: {t.get('answer','')}\nScore: {t.get('score','?')}/10"
        for i, t in enumerate(transcript)
    ])

    prompt = f"""You are a senior hiring manager reviewing a completed {job_role} interview.

Full Q&A Transcript:
{qa_text}

Return ONLY this JSON (no markdown, no extra text):
{{
  "overall_score": <integer 0-100>,
  "technical_score": <integer 0-100>,
  "communication_score": <integer 0-100>,
  "summary": "<3-4 sentence overall assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<area to improve 1>", "<area to improve 2>"],
  "recommendation": "<Strong Hire | Hire | Maybe | No Hire>"
}}"""

    raw    = gemini_ask(current_app._get_current_object(), prompt)
    result = parse_json_response(raw)

    if result and 'overall_score' in result:
        overall = result['overall_score']
        return jsonify({
            'success':             True,
            'overall_score':       overall,
            'technical_score':     result.get('technical_score', overall),
            'communication_score': result.get('communication_score', overall),
            'passed':              overall >= pass_threshold,
            'pass_threshold':      pass_threshold,
            'recommendation':      result.get('recommendation', 'Maybe'),
            'summary':             result.get('summary', ''),
            'strengths':           result.get('strengths', []),
            'improvements':        result.get('improvements', []),
        })

    avg     = sum(t.get('score', 5) for t in transcript) / max(len(transcript), 1)
    overall = int(avg * 10)
    rec     = ('Strong Hire' if overall >= 80 else
               'Hire'        if overall >= 65 else
               'Maybe'       if overall >= 50 else 'No Hire')

    return jsonify({
        'success':             True,
        'overall_score':       overall,
        'technical_score':     overall,
        'communication_score': overall,
        'passed':              overall >= pass_threshold,
        'pass_threshold':      pass_threshold,
        'recommendation':      rec,
        'summary':             f'Average answer score: {avg:.1f}/10.',
        'strengths':           ['Completed the full interview'],
        'improvements':        ['More detailed answers would improve the score'],
    })