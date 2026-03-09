import os
from utils.helpers import parse_json_response


def get_gemini_client(app):
    import google.generativeai as genai
    api_key = app.config.get('GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')


def gemini_ask(app, prompt):
    """Send a prompt to Gemini and return the text response. Returns None on failure."""
    model = get_gemini_client(app)
    if not model:
        return None
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None
