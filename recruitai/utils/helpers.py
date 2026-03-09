import random
import string
import json


def make_room_code(n=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def get_violation_info(vtype):
    return {
        'tab_switch':       {'severity': 2, 'description': 'User switched tab/window'},
        'exit_fullscreen':  {'severity': 3, 'description': 'Exited fullscreen'},
        'no_face':          {'severity': 2, 'description': 'Face not visible'},
        'multiple_faces':   {'severity': 3, 'description': 'Multiple faces detected'},
        'right_click':      {'severity': 1, 'description': 'Right-click attempt'},
        'copy_attempt':     {'severity': 1, 'description': 'Copy attempt'},
        'paste_attempt':    {'severity': 1, 'description': 'Paste attempt'},
        'devtools':         {'severity': 2, 'description': 'DevTools attempt'},
        'gaze_away':        {'severity': 2, 'description': 'Looking away from screen'},
        'phone_detected':   {'severity': 3, 'description': 'Phone/device detected'},
        'device_detected':  {'severity': 3, 'description': 'Unauthorized device detected'},
        'second_screen':    {'severity': 3, 'description': 'Second screen detected'},
    }.get(vtype, {'severity': 1, 'description': 'Unknown violation'})


def parse_json_response(raw):
    """Strip markdown code fences and parse JSON safely."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith('```'):
        parts = text.split('```')
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def make_google_calendar_link(title, description, start_dt, end_dt=None):
    """Generates a Google Calendar event creation link."""
    import urllib.parse
    from datetime import timedelta
    if end_dt is None:
        end_dt = start_dt + timedelta(hours=1)
    fmt = '%Y%m%dT%H%M%SZ'
    params = {
        'action': 'TEMPLATE',
        'text': title,
        'details': description,
        'dates': f"{start_dt.strftime(fmt)}/{end_dt.strftime(fmt)}",
    }
    return 'https://calendar.google.com/calendar/render?' + urllib.parse.urlencode(params)
