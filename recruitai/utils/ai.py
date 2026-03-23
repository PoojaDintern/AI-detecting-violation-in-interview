import os
from utils.helpers import parse_json_response


def get_gemini_client(app):
    import google.generativeai as genai
    api_key = app.config.get('GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')


def gemini_ask(app, prompt, track_type=None, track_label='',
               recruiter_id=None, session_id=None):
    """Send a prompt to Gemini and return the text response.
    Saves token usage to DB if track_type is provided.

    track_type values:
        'questions'   — AI interview question generation
        'evaluations' — Per-answer evaluation
        'final'       — Final transcript evaluation
        'mcq_questions' — MCQ question generation
    """
    model = get_gemini_client(app)
    if not model:
        return None
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # ── Save token usage to DB ──────────────────────────────
        if track_type:
            try:
                _save_token_usage(
                    app           = app,
                    call_type     = track_type,
                    label         = track_label,
                    prompt        = prompt,
                    response_text = text,
                    response_obj  = response,
                    recruiter_id  = recruiter_id,
                    session_id    = session_id,
                )
            except Exception as te:
                print(f'[TokenUsage] Failed to save: {te}')

        return text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def _save_token_usage(app, call_type, label, prompt, response_text,
                      response_obj, recruiter_id=None, session_id=None):
    """Save a token usage record to the database."""
    from extensions import db
    from models.token_usage import TokenUsage

    # Get token counts from Gemini metadata
    meta            = getattr(response_obj, 'usage_metadata', None)
    prompt_tokens   = getattr(meta, 'prompt_token_count',     0) if meta else 0
    response_tokens = getattr(meta, 'candidates_token_count', 0) if meta else 0
    total_tokens    = getattr(meta, 'total_token_count',       0) if meta else 0

    # Fallback estimate if metadata not available (~4 chars per token)
    if not total_tokens:
        prompt_tokens   = len(prompt) // 4
        response_tokens = len(response_text) // 4
        total_tokens    = prompt_tokens + response_tokens

    record = TokenUsage(
        recruiter_id    = recruiter_id,
        session_id      = session_id,
        call_type       = call_type,
        label           = str(label)[:150] if label else '',
        prompt_tokens   = prompt_tokens,
        response_tokens = response_tokens,
        total_tokens    = total_tokens,
        prompt_preview  = str(prompt)[:300],
    )

    with app.app_context():
        db.session.add(record)
        db.session.commit()