from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required
from models import ExamQuestion
from utils.ai import gemini_ask
from utils.helpers import parse_json_response

ai_bp = Blueprint('ai', __name__)


@ai_bp.route('/api/ai-generate-questions', methods=['POST'])
@login_required
def ai_generate_questions():
    data     = request.get_json() or {}
    job_role = data.get('job_role', 'Software Developer')

    prompt = f"""You are an expert technical recruiter interviewing a candidate for a {job_role} position.

Generate exactly 10 interview questions. Return ONLY a valid JSON array like this:
[
  {{
    "id": 1,
    "question": "Your interview question here?",
    "type": "technical",
    "expected_keywords": ["keyword1", "keyword2", "keyword3"],
    "follow_up": "A follow-up question if they answer correctly?"
  }}
]

Rules:
- Mix of difficulty: 3 easy, 4 medium, 3 hard
- Cover fundamentals, problem-solving, best practices, real-world scenarios
- Questions must be specific to {job_role}
- Return ONLY the JSON array. No explanation, no markdown, no extra text."""

    raw       = gemini_ask(current_app._get_current_object(), prompt)
    questions = parse_json_response(raw)

    if questions and isinstance(questions, list) and len(questions) > 0:
        return jsonify({'success': True, 'source': 'gemini', 'questions': questions})

    print("⚠️  Gemini unavailable — falling back to DB questions")
    qs = ExamQuestion.query.filter_by(job_role=job_role).limit(10).all()
    if not qs:
        qs = ExamQuestion.query.limit(10).all()
    return jsonify({'success': True, 'source': 'db_fallback',
                    'questions': [q.to_dict() for q in qs]})


@ai_bp.route('/api/ai-evaluate-answer', methods=['POST'])
@login_required
def ai_evaluate_answer():
    data     = request.get_json() or {}
    question = data.get('question', '')
    answer   = data.get('answer', '')
    job_role = data.get('job_role', 'Developer')
    keywords = data.get('expected_keywords', [])

    prompt = f"""You are evaluating a {job_role} interview answer.

Question: {question}
Candidate's Answer: {answer}
Expected Keywords/Concepts: {', '.join(keywords) if keywords else 'N/A'}

Evaluate strictly and fairly. Return ONLY this JSON object (no markdown, no explanation):
{{
  "score": <integer 1-10>,
  "feedback": "<exactly 2 sentences of constructive feedback>",
  "follow_up": "<one follow-up question if the answer was incomplete, or null if complete>",
  "strong_points": ["<what they got right>"],
  "missing_points": ["<what was missing>"]
}}"""

    raw    = gemini_ask(current_app._get_current_object(), prompt)
    result = parse_json_response(raw)

    if result and 'score' in result:
        return jsonify({'success': True, **result})

    print("⚠️  Gemini unavailable — using keyword fallback scoring")
    matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
    score   = min(10, max(1, 4 + round((matches / max(len(keywords), 1)) * 6)))
    return jsonify({'success': True, 'score': score,
                    'feedback': f'Answer recorded. {matches} of {len(keywords)} key concepts mentioned.',
                    'follow_up': None, 'strong_points': [], 'missing_points': []})


@ai_bp.route('/api/ai-final-evaluation', methods=['POST'])
@login_required
def ai_final_evaluation():
    data       = request.get_json() or {}
    transcript = data.get('transcript', [])
    job_role   = data.get('job_role', 'Developer')

    if not transcript:
        return jsonify({'success': True, 'overall_score': 50,
                        'summary': 'No transcript available.', 'recommendation': 'Review required.'})

    qa_text = '\n\n'.join([
        f"Q{i+1}: {t.get('question','')}\nAnswer: {t.get('answer','')}\nScore: {t.get('score','?')}/10"
        for i, t in enumerate(transcript)
    ])

    prompt = f"""You are a senior hiring manager reviewing a completed {job_role} interview.

Full Q&A Transcript:
{qa_text}

Based on all answers, provide a final evaluation. Return ONLY this JSON (no markdown, no extra text):
{{
  "overall_score": <integer 0-100>,
  "technical_score": <integer 0-100>,
  "communication_score": <integer 0-100>,
  "summary": "<3-4 sentence overall assessment of the candidate>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<area to improve 1>", "<area to improve 2>"],
  "recommendation": "<exactly one of: Strong Hire | Hire | Maybe | No Hire>"
}}"""

    raw    = gemini_ask(current_app._get_current_object(), prompt)
    result = parse_json_response(raw)

    if result and 'overall_score' in result:
        return jsonify({'success': True, **result})

    print("⚠️  Gemini unavailable — using average score fallback")
    avg     = sum(t.get('score', 5) for t in transcript) / max(len(transcript), 1)
    overall = int(avg * 10)
    rec     = 'Strong Hire' if overall >= 80 else 'Hire' if overall >= 65 else 'Maybe' if overall >= 50 else 'No Hire'
    return jsonify({'success': True, 'overall_score': overall,
                    'technical_score': overall, 'communication_score': overall,
                    'summary': f'Interview completed with an average score of {avg:.1f}/10.',
                    'strengths': ['Completed the interview'], 'improvements': ['More detail needed'],
                    'recommendation': rec})
