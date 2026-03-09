import json
from io import BytesIO
from flask import Blueprint, jsonify, send_file
from flask_login import login_required, current_user
from extensions import db
from models import TestSubmission, Violation, GazeEvent, DeviceAlert
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

results_bp = Blueprint('results', __name__)


@results_bp.route('/api/results/<int:submission_id>')
@login_required
def get_results(submission_id):
    sub = db.session.get(TestSubmission, submission_id)
    if not sub:
        return jsonify({"success": False, "message": "Submission not found"}), 404
    if sub.user_id != current_user.id and not current_user.is_admin and not current_user.is_recruiter:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    violations  = Violation.query.filter_by(session_id=sub.session_id).order_by(Violation.timestamp).all()     if sub.session_id else []
    gaze_objs   = GazeEvent.query.filter_by(session_id=sub.session_id).order_by(GazeEvent.timestamp).all()     if sub.session_id else []
    device_objs = DeviceAlert.query.filter_by(session_id=sub.session_id).order_by(DeviceAlert.timestamp).all() if sub.session_id else []

    breakdown = {}
    for v in violations:
        breakdown[v.violation_type] = breakdown.get(v.violation_type, 0) + 1

    ai_feedback = {}
    if sub.ai_feedback:
        try:
            ai_feedback = json.loads(sub.ai_feedback)
        except Exception:
            pass

    gaze_list   = [{'timestamp': g.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': g.direction,
                    'confidence': round(g.confidence or 0, 2)} for g in gaze_objs]
    device_list = [{'timestamp': d.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'device_type': d.device_type,
                    'confidence': round(d.confidence or 0, 2)} for d in device_objs]

    return jsonify({'success': True, 'submission': sub.to_dict(),
                    'violations':    [v.to_dict() for v in violations],
                    'breakdown':     breakdown,
                    'gaze_events':   len(gaze_list),
                    'gaze_list':     gaze_list,
                    'device_alerts': len(device_list),
                    'device_list':   device_list,
                    'ai_feedback':   ai_feedback})


@results_bp.route('/api/my-submissions')
@login_required
def my_submissions():
    subs = TestSubmission.query.filter_by(user_id=current_user.id).order_by(TestSubmission.submitted_at.desc()).all()
    return jsonify({'success': True, 'submissions': [s.to_dict() for s in subs]})


@results_bp.route('/download-report/<int:submission_id>')
@login_required
def download_report(submission_id):
    sub = db.session.get(TestSubmission, submission_id)
    if not sub:
        return "Not Found", 404
    if sub.user_id != current_user.id and not current_user.is_admin and not current_user.is_recruiter:
        return "Unauthorized", 403

    buf = BytesIO()
    pdf = pdf_canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(1*inch, h-1*inch, "AI Recruiting System — Interview Report")
    pdf.setFont("Helvetica", 12)
    y = h - 1.6*inch
    for line in [f"Candidate: {sub.user.full_name} ({sub.user.username})",
                 f"Job Role: {sub.job_role}", f"Mode: {sub.mode.replace('_',' ').title()}",
                 f"Submitted: {sub.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Attempt: #{sub.attempt_number}"]:
        pdf.drawString(1*inch, y, line); y -= 0.28*inch
    y -= 0.2*inch
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(1*inch, y, f"Credibility Score:  {sub.credibility_score} / 100"); y -= 0.3*inch
    pdf.drawString(1*inch, y, f"Interview Score:    {sub.interview_score} / 100"); y -= 0.35*inch
    pdf.setFillColor(colors.green if sub.passed else colors.red)
    pdf.drawString(1*inch, y, f"Status: {'PASSED ✓' if sub.passed else 'FAILED ✗'}")
    pdf.setFillColor(colors.black)
    y -= 0.5*inch
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(1*inch, y, f"Total Violations: {sub.total_violations}")
    violations = Violation.query.filter_by(session_id=sub.session_id).all()
    y -= 0.3*inch
    pdf.setFont("Helvetica", 10)
    for v in violations[:20]:
        y -= 0.22*inch
        if y < 1*inch:
            pdf.showPage(); y = h - 1*inch
        sev_label = {1: 'Low', 2: 'Medium', 3: 'High'}.get(v.severity, '')
        pdf.drawString(1*inch, y,
            f"  • {v.timestamp.strftime('%H:%M:%S')}  {v.violation_type}  [{sev_label}]")

    if sub.ai_feedback:
        try:
            fb = json.loads(sub.ai_feedback)
            y -= 0.5*inch
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(1*inch, y, "AI Interview Feedback:"); y -= 0.25*inch
            pdf.setFont("Helvetica", 10)
            summary = fb.get('summary', '')
            for chunk in [summary[i:i+90] for i in range(0, len(summary), 90)]:
                y -= 0.22*inch
                if y < 1*inch:
                    pdf.showPage(); y = h - 1*inch
                pdf.drawString(1*inch, y, chunk)
        except Exception:
            pass

    pdf.setFont("Helvetica", 8)
    pdf.drawString(1*inch, 0.5*inch, "Generated by AI Recruiting & Proctoring System (Gemini Edition)")
    pdf.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'report_{sub.user.username}_{sub.id}.pdf',
                     mimetype='application/pdf')
