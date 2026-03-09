import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_interview_email(to_email, to_name, recruiter_name, company_name,
                         job_role, scheduled_at_str, room_code, calendar_link,
                         role='candidate', smtp_user=None, smtp_pass=None):
    smtp_user = smtp_user or os.environ.get('EMAIL_USER', '')
    smtp_pass = smtp_pass or os.environ.get('EMAIL_PASS', '')
    base_url = os.environ.get("BASE_URL", "http://localhost:5000")

    candidate_join_url = f"{base_url}/interview_room.html?room={room_code}"
    recruiter_join_url = f"{base_url}/recruiter_room.html?room={room_code}"

    calendar_btn = (f'<div style="text-align:center;margin:12px 0;">'
                    f'<a href="{calendar_link}" style="color:#4f8ef7;font-size:13px;text-decoration:none;">'
                    f'📅 Add to Google Calendar</a></div>') if calendar_link else ''

    subject = f"RecruitAI — Interview Scheduled: {job_role} at {company_name}"

    def _header(subtitle):
        return f"""
<div style="background:#0f1117;padding:24px;border-radius:10px;text-align:center;margin-bottom:24px;">
  <h1 style="color:#4f8ef7;margin:0;font-size:26px;">🤖 RecruitAI</h1>
  <p style="color:#7a88a8;margin:8px 0 0;font-size:13px;">{subtitle}</p>
</div>"""

    def _info_table(rows):
        trs = ''.join(
            f'<tr><td style="padding:11px 14px;background:#e8edf5;font-weight:600;width:38%;font-size:13px;">{k}</td>'
            f'<td style="padding:11px 14px;background:#fff;font-size:13px;">{v}</td></tr>'
            for k, v in rows
        )
        return f'<table style="width:100%;margin:20px 0;border-collapse:collapse;border-radius:8px;overflow:hidden;">{trs}</table>'

    def _join_btn(url, label, color="#4f8ef7"):
        return f"""
<div style="text-align:center;margin:28px 0 16px;">
  <a href="{url}" style="background:{color};color:#ffffff;padding:16px 40px;border-radius:10px;
     text-decoration:none;font-weight:700;font-size:16px;display:inline-block;">
    {label}
  </a>
</div>"""

    def _footer():
        return '<p style="color:#aaa;font-size:11px;text-align:center;margin-top:32px;border-top:1px solid #e8edf5;padding-top:16px;">RecruitAI — AI-Powered Recruitment Platform</p>'

    if role == 'candidate':
        rows = [
            ("📋 Position",      job_role),
            ("🏢 Company",       company_name),
            ("👤 Recruiter",     recruiter_name),
            ("🗓️ Date & Time",   f"<strong>{scheduled_at_str}</strong>"),
            ("🔑 Room Code",     f'<code style="background:#0f1117;color:#4f8ef7;padding:4px 10px;border-radius:5px;">{room_code}</code>'),
        ]
        body_html = f"""
<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#f9fafb;padding:32px;border-radius:14px;">
  {_header("Interview Invitation")}
  <h2 style="color:#1a2035;">Hello, {to_name}! 👋</h2>
  <p style="color:#555;line-height:1.7;">Your interview has been <strong>scheduled</strong>. Please be ready on time.</p>
  {_info_table(rows)}
  {_join_btn(candidate_join_url, "🚀 Join My Interview →", "#4f8ef7")}
  {calendar_btn}
  {_footer()}
</div>"""
    else:
        recruiter_room_url = f"{base_url}/recruiter_room.html?room={room_code}"
        rows = [
            ("👤 Candidate",    to_name),
            ("📋 Position",     job_role),
            ("🏢 Company",      company_name),
            ("🗓️ Date & Time",  f"<strong>{scheduled_at_str}</strong>"),
            ("🔑 Room Code",    f'<code style="background:#0f1117;color:#4f8ef7;padding:4px 10px;border-radius:5px;">{room_code}</code>'),
        ]
        body_html = f"""
<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#f9fafb;padding:32px;border-radius:14px;">
  {_header("Interview Scheduled — Recruiter Copy")}
  <h2 style="color:#1a2035;">Interview Confirmed ✅</h2>
  {_info_table(rows)}
  {_join_btn(recruiter_room_url, "🎙 Join Recruiter Room →", "#059669")}
  {calendar_btn}
  {_footer()}
</div>"""

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = smtp_user or 'noreply@recruitai.com'
    msg['To']      = to_email
    msg.attach(MIMEText(body_html, 'html'))

    if smtp_user and smtp_pass:
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, [to_email], msg.as_string())
            print(f"✅ Email sent to {to_email} ({role})")
        except Exception as e:
            print(f"⚠️  Email failed to {to_email}: {e}")
    else:
        print(f"📧 [No SMTP] Would send to {to_email}: {subject}")
    return True


def send_interviewer_assignment_email(to_email, to_name, recruiter_name, candidate_name,
                                       job_role, round_name, room_code, portal_url, room_url,
                                       notes='', recruiter_smtp=None, recruiter_pass=None):
    smtp_user = recruiter_smtp or os.environ.get('EMAIL_USER', '')
    smtp_pass = recruiter_pass or os.environ.get('EMAIL_PASS', '')
    if not smtp_user or not smtp_pass:
        return

    subject = f"RecruitAI — You've been assigned as Interviewer: {job_role} ({round_name})"
    body_html = f"""
<div style="font-family:Arial,sans-serif;max-width:580px;margin:0 auto;background:#f9fafb;padding:32px;border-radius:12px;">
  <div style="background:#0f1117;padding:24px;border-radius:10px;text-align:center;margin-bottom:24px;">
    <h1 style="color:#4361ee;margin:0;">🤖 RecruitAI</h1>
    <p style="color:#7a88a8;margin:8px 0 0;">Interviewer Assignment</p>
  </div>
  <h2 style="color:#1a2035;">Hello, {to_name}!</h2>
  <p style="color:#444;line-height:1.6;">You've been assigned as an <strong>interviewer</strong> by <strong>{recruiter_name}</strong>.</p>
  <table style="width:100%;margin:20px 0;border-collapse:collapse;">
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;width:40%;">Candidate</td><td style="padding:10px 14px;background:#fff;">{candidate_name}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Position</td><td style="padding:10px 14px;background:#fff;">{job_role}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Round</td><td style="padding:10px 14px;background:#fff;">{round_name}</td></tr>
    <tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Room Code</td><td style="padding:10px 14px;background:#fff;"><code style="background:#0f1117;color:#4361ee;padding:4px 8px;border-radius:4px;">{room_code}</code></td></tr>
    {f'<tr><td style="padding:10px 14px;background:#e8edf5;font-weight:600;">Notes</td><td style="padding:10px 14px;background:#fff;">{notes}</td></tr>' if notes else ''}
  </table>
  <div style="text-align:center;margin:24px 0 12px;">
    <a href="{room_url}" style="background:#4361ee;color:#fff;padding:14px 36px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px;display:inline-block;">🚀 Join Interview Room →</a>
  </div>
  <div style="text-align:center;margin-bottom:24px;">
    <a href="{portal_url}" style="color:#4361ee;font-size:13px;text-decoration:underline;">📋 View Assignment &amp; Submit Evaluation</a>
  </div>
  <p style="color:#aaa;font-size:12px;text-align:center;margin-top:32px;">RecruitAI — AI-Powered Recruitment Platform</p>
</div>"""

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = smtp_user
    msg['To']      = to_email
    msg.attach(MIMEText(body_html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg.as_string())
    except Exception as e:
        print(f'[EMAIL] Interviewer assignment email failed: {e}')
