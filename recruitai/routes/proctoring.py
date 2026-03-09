import cv2
import numpy as np
import base64
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from extensions import db, socketio
from models import InterviewSession, Violation, GazeEvent, DeviceAlert, User
from utils.proctoring import detect_faces, eye_cascade

proctoring_bp = Blueprint('proctoring', __name__)

# Per-session consecutive miss counters
_face_miss_counter = {}
_gaze_away_counter = {}


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


def calc_credibility(session_id):
    DEVICE_VIOLATION_TYPES = {'phone_detected', 'device_detected'}
    violations = Violation.query.filter_by(session_id=session_id).all()
    scored = [v for v in violations if v.violation_type not in DEVICE_VIOLATION_TYPES]
    severity_points = current_app.config.get('SEVERITY_POINTS', {1: 5, 2: 10, 3: 15})
    score = 100
    for v in scored:
        score -= severity_points.get(v.severity, 5)
    if len(scored) > 10:
        score -= (len(scored) - 10) * 2
    return max(0, min(100, score))


def log_violation_db(user_id, session_id, vtype, gaze_data=None, device_data=None):
    info = get_violation_info(vtype)
    v = Violation(
        user_id=user_id, session_id=session_id,
        violation_type=vtype, severity=info['severity'],
        description=info['description'],
        gaze_data=json.dumps(gaze_data) if gaze_data else None,
        device_data=json.dumps(device_data) if device_data else None,
    )
    db.session.add(v)
    db.session.commit()
    sess = db.session.get(InterviewSession, session_id) if session_id else None
    new_score = calc_credibility(session_id) if session_id else 100
    if sess:
        sess.credibility_score = new_score
        db.session.commit()
    alert_data = {
        'user_id': user_id,
        'username': db.session.get(User, user_id).username,
        'violation_type': vtype, 'severity': info['severity'],
        'description': info['description'],
        'timestamp': datetime.utcnow().strftime('%H:%M:%S'),
        'new_credibility': new_score,
        'session_id': session_id,
    }
    socketio.emit('violation_alert', alert_data, room='dashboard')
    if sess and sess.room_code:
        socketio.emit('violation_alert', alert_data, room=sess.room_code)
    return v


@proctoring_bp.route('/api/room-scan', methods=['POST'])
@login_required
def room_scan():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        frames     = data.get('frames', [])
        if not frames:
            return jsonify({'success': True, 'issues': []})

        issues = []
        multiple_seen = 0
        device_seen   = 0
        no_face_count = 0

        for idx, frame_b64 in enumerate(frames):
            try:
                raw   = frame_b64.split(',')[1] if ',' in frame_b64 else frame_b64
                buf   = np.frombuffer(base64.b64decode(raw), np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                faces = detect_faces(frame, conf_threshold=0.50)
                nf = len(faces)
                if nf > 1:
                    multiple_seen += 1
                elif nf == 0 and idx < 2:
                    no_face_count += 1

                h, w = frame.shape[:2]
                hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_b = np.array([100, 50, 50]); upper_b = np.array([130, 255, 255])
                mask = cv2.inRange(hsv, lower_b, upper_b)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 2000: continue
                    x2, y2, cw, ch = cv2.boundingRect(c)
                    ratio = float(max(cw, ch)) / float(min(cw, ch) + 1)
                    frame_ratio = area / float(w * h)
                    if 1.3 < ratio < 3.0 and frame_ratio < 0.35:
                        device_seen += 1
            except Exception as fe:
                print(f"Frame {idx} error: {fe}")
                continue

        if multiple_seen >= 2:
            issues.append("multiple people detected in room")
            log_violation_db(current_user.id, session_id, "multiple_faces",
                             gaze_data=f"Room scan: multiple people in {multiple_seen} frames")
        if device_seen >= 2:
            issues.append("prohibited device visible")
            log_violation_db(current_user.id, session_id, "device_detected",
                             device_data=f"Room scan: device in {device_seen} frames")
        if no_face_count >= 2:
            issues.append("candidate not visible at scan start")

        return jsonify({'success': True, 'issues': issues, 'frames_analysed': len(frames)})
    except Exception as e:
        print(f"Room scan error: {e}")
        return jsonify({'success': True, 'issues': []})


@proctoring_bp.route('/detect-face', methods=['POST'])
@login_required
def detect_face():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        img_b64    = data['image'].split(',')[1]
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

        faces = detect_faces(frame, conf_threshold=0.55)
        n     = len(faces)

        should_log = False
        if session_id:
            sess = db.session.get(InterviewSession, session_id)
            if sess and sess.status == 'active' and sess.started_at:
                elapsed = (datetime.utcnow() - sess.started_at).total_seconds()
                if elapsed >= 15:
                    should_log = True

        if should_log:
            key = f'no_face_{session_id}'
            _face_miss_counter[key] = _face_miss_counter.get(key, 0) + (1 if n == 0 else -2)
            _face_miss_counter[key] = max(0, _face_miss_counter[key])

            if n == 0 and _face_miss_counter[key] >= 2:
                log_violation_db(current_user.id, session_id, 'no_face')
                _face_miss_counter[key] = 0
            elif n > 1:
                log_violation_db(current_user.id, session_id, 'multiple_faces')

        return jsonify({'success': True, 'face_detected': n >= 1, 'num_faces': n})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@proctoring_bp.route('/api/check-head-pose', methods=['POST'])
@login_required
def check_head_pose():
    try:
        data          = request.get_json() or {}
        required_pose = data.get('required_pose')
        img_b64       = data.get('image', '')
        if not img_b64 or not required_pose:
            return jsonify({'success': True, 'pose_detected': True})

        img_bytes = base64.b64decode(img_b64.split(',')[-1])
        frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': True, 'pose_detected': True})

        fh, fw = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray   = cv2.equalizeHist(gray)
        faces  = detect_faces(frame, conf_threshold=0.45)

        if len(faces) == 0:
            pose_ok = (required_pose == 'down')
            return jsonify({'success': True, 'pose_detected': pose_ok, 'reason': 'no_face_detected'})

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx_ratio = (x + w / 2) / fw
        cy_ratio = (y + h / 2) / fh
        face_aspect = w / max(h, 1)
        is_profile  = face_aspect < 0.62
        left_zone   = cx_ratio < 0.42
        right_zone  = cx_ratio > 0.58
        centre_zone = 0.35 < cx_ratio < 0.65

        eyes_high = False
        face_roi  = gray[y:y+h, x:x+w]
        eyes      = eye_cascade.detectMultiScale(face_roi, 1.1, 6, minSize=(15, 15))
        if len(eyes) >= 1:
            avg_eye_y = sum(ey + eh / 2 for (_, ey, _, eh) in eyes) / len(eyes)
            eyes_high = (avg_eye_y / max(h, 1)) < 0.38

        pose_ok = False
        reason  = ''
        if required_pose == 'center':
            pose_ok = centre_zone and not is_profile
            reason  = f'cx={cx_ratio:.2f} aspect={face_aspect:.2f}'
        elif required_pose == 'left':
            pose_ok = right_zone or is_profile
            reason  = f'cx={cx_ratio:.2f} profile={is_profile}'
        elif required_pose == 'right':
            pose_ok = left_zone or is_profile
            reason  = f'cx={cx_ratio:.2f} profile={is_profile}'
        elif required_pose == 'down':
            pose_ok = (cy_ratio > 0.55) or not eyes_high
            reason  = f'cy={cy_ratio:.2f}'

        return jsonify({'success': True, 'pose_detected': pose_ok, 'reason': reason,
                        'cx_ratio': round(cx_ratio, 3), 'face_aspect': round(face_aspect, 3),
                        'is_profile': is_profile})
    except Exception as e:
        print(f'check_head_pose error: {e}')
        return jsonify({'success': True, 'pose_detected': True})


@proctoring_bp.route('/analyze-gaze', methods=['POST'])
@login_required
def analyze_gaze():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        img_b64    = data['image'].split(',')[1]
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw     = frame.shape[:2]

        gaze_result = {'direction': 'unknown', 'confidence': 0.0, 'looking_away': False}
        faces = detect_faces(frame, conf_threshold=0.50)

        if len(faces) == 1:
            fx, fy, fw_f, fh_f = faces[0]
            face_roi_gray = gray[fy:fy+fh_f, fx:fx+fw_f]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            face_roi_eq = clahe.apply(face_roi_gray)
            eyes = eye_cascade.detectMultiScale(face_roi_eq, scaleFactor=1.05,
                                                 minNeighbors=6, minSize=(18, 18))
            eyes = [e for e in eyes if e[1] < int(fh_f * 0.65)]

            direction, looking_away, confidence = 'center', False, 0.0

            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]

                def pupil_center(roi):
                    blurred = cv2.GaussianBlur(roi, (7, 7), 0)
                    thresh  = cv2.adaptiveThreshold(blurred, 255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, 11, 2)
                    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    m = cv2.moments(thresh)
                    if m['m00'] > 0:
                        return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
                    return roi.shape[1] // 2, roi.shape[0] // 2

                p1x, p1y = pupil_center(face_roi_eq[ey1:ey1+eh1, ex1:ex1+ew1])
                p2x, p2y = pupil_center(face_roi_eq[ey2:ey2+eh2, ex2:ex2+ew2])

                avg_h_ratio = ((p1x / max(ew1, 1)) + (p2x / max(ew2, 1))) / 2
                avg_v_ratio = ((p1y / max(eh1, 1)) + (p2y / max(eh2, 1))) / 2

                if avg_h_ratio < 0.30:
                    direction, looking_away = 'left', True
                elif avg_h_ratio > 0.70:
                    direction, looking_away = 'right', True
                elif avg_v_ratio < 0.28:
                    direction, looking_away = 'up', True
                elif avg_v_ratio > 0.78:
                    direction, looking_away = 'down', True
                else:
                    direction, looking_away = 'center', False

                confidence = 0.88
                gaze_result = {'direction': direction, 'confidence': confidence,
                               'looking_away': looking_away,
                               'h_ratio': round(avg_h_ratio, 3),
                               'v_ratio': round(avg_v_ratio, 3)}
            elif len(eyes) == 1:
                ex1, ey1, ew1, eh1 = eyes[0]
                h_ratio = ew1 // 2 / max(ew1, 1)
                if h_ratio < 0.28:
                    direction, looking_away = 'left', True
                elif h_ratio > 0.72:
                    direction, looking_away = 'right', True
                else:
                    direction, looking_away = 'center', False
                confidence = 0.55
                gaze_result = {'direction': direction, 'confidence': confidence, 'looking_away': looking_away}
            else:
                face_cx_ratio = (fx + fw_f / 2) / fw
                face_cy_ratio = (fy + fh_f / 2) / fh
                aspect        = fw_f / max(fh_f, 1)

                if aspect < 0.60:
                    direction = 'left' if face_cx_ratio < 0.5 else 'right'
                    looking_away = True
                elif face_cx_ratio < 0.25:
                    direction, looking_away = 'left', True
                elif face_cx_ratio > 0.75:
                    direction, looking_away = 'right', True
                elif face_cy_ratio < 0.20:
                    direction, looking_away = 'up', True
                elif face_cy_ratio > 0.80:
                    direction, looking_away = 'down', True
                else:
                    direction, looking_away = 'center', False

                confidence = 0.60
                gaze_result = {'direction': direction if looking_away else 'center',
                               'confidence': confidence, 'looking_away': looking_away,
                               'method': 'head_pose'}

        elif len(faces) == 0:
            gaze_result = {'direction': 'no_face', 'confidence': 0.5, 'looking_away': True}

        # Violation logging with cooldown
        should_log_gaze = False
        if session_id:
            sess_g = db.session.get(InterviewSession, session_id)
            if sess_g and sess_g.status == 'active' and sess_g.started_at:
                if (datetime.utcnow() - sess_g.started_at).total_seconds() >= 15:
                    should_log_gaze = True

        if should_log_gaze and gaze_result.get('looking_away'):
            direction = gaze_result.get('direction', 'unknown')
            gk = f'gaze_{session_id}'
            _gaze_away_counter[gk] = _gaze_away_counter.get(gk, 0) + 1
            if _gaze_away_counter[gk] >= 2:
                ge = GazeEvent(user_id=current_user.id, session_id=session_id,
                               direction=direction, confidence=gaze_result.get('confidence', 0.5))
                db.session.add(ge)
                db.session.commit()
                log_violation_db(current_user.id, session_id, 'gaze_away',
                                 gaze_data={'direction': direction,
                                            'confidence': round(gaze_result.get('confidence', 0), 3),
                                            'method': gaze_result.get('method', 'pupil')})
                _gaze_away_counter[gk] = 0
        else:
            gk = f'gaze_{session_id}'
            if gk in _gaze_away_counter:
                _gaze_away_counter[gk] = 0

        return jsonify({'success': True, 'gaze': gaze_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@proctoring_bp.route('/detect-device', methods=['POST'])
@login_required
def detect_device():
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        img_b64    = data['image'].split(',')[1]
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        fh, fw = frame.shape[:2]
        frame_area = fw * fh

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq_gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(eq_gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges   = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        faces        = detect_faces(frame, conf_threshold=0.55)
        face_regions = [(x, y, x+w, y+h) for (x, y, w, h) in faces]

        def in_face(cx, cy):
            return any(fx1 < cx < fx2 and fy1 < cy < fy2 for (fx1, fy1, fx2, fy2) in face_regions)

        hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        screen_mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        screen_cnts, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected    = False
        confidence  = 0.0
        device_type = None

        PROFILES = [
            ('phone',     0.006, 0.20,  1.7, 3.2,  0.82),
            ('tablet',    0.12,  0.40,  1.3, 1.8,  0.82),
            ('laptop',    0.15,  0.55,  2.4, 4.2,  0.81),
            ('earphones', 0.001, 0.008, 0.7, 1.5,  0.80),
        ]
        MIN_DETECT_CONF = 0.84

        def check_rect(x, y, w, h, area):
            nonlocal detected, confidence, device_type
            aspect   = max(w, h) / max(min(w, h), 1)
            area_rat = area / frame_area
            cx, cy   = x + w // 2, y + h // 2
            if in_face(cx, cy):
                return
            for label, min_ar, max_ar, min_asp, max_asp, base_c in PROFILES:
                if min_ar <= area_rat <= max_ar and min_asp <= aspect <= max_asp:
                    fill = (area_rat - min_ar) / max(max_ar - min_ar, 1e-6)
                    conf = min(0.95, base_c + fill * 0.18)
                    if conf > confidence:
                        confidence, device_type = conf, label
                        detected = conf >= MIN_DETECT_CONF

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frame_area * 0.0004: continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                check_rect(x, y, w, h, area)

        for cnt in screen_cnts:
            area = cv2.contourArea(cnt)
            if area < frame_area * 0.025: continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect   = max(w, h) / max(min(w, h), 1)
            area_rat = area / frame_area
            cx, cy   = x + w // 2, y + h // 2
            if in_face(cx, cy): continue
            if 1.6 <= aspect <= 3.8 and 0.05 <= area_rat <= 0.45:
                label = 'laptop' if aspect >= 2.4 else 'tablet'
                conf  = min(0.92, 0.78 + area_rat * 0.20)
                if conf > confidence and conf >= MIN_DETECT_CONF:
                    confidence, device_type, detected = conf, label, True

        ear_zones = [(0, 0, fw//3, int(fh*0.65)), (fw*2//3, 0, fw, int(fh*0.65))]
        for (ex1, ey1, ex2, ey2) in ear_zones:
            roi   = blurred[ey1:ey2, ex1:ex2]
            r_edg = cv2.Canny(roi, 20, 70)
            r_edg = cv2.dilate(r_edg, kernel, iterations=1)
            ecnts, _ = cv2.findContours(r_edg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in ecnts:
                area = cv2.contourArea(cnt)
                zone_area = max((ex2-ex1)*(ey2-ey1), 1)
                ea_ratio  = area / zone_area
                if not 0.003 <= ea_ratio <= 0.15: continue
                peri = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / max(peri ** 2, 1)
                aprx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if circ > 0.25 or 4 <= len(aprx) <= 8:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cx2, cy2 = ex1+bx+bw//2, ey1+by+bh//2
                    if not in_face(cx2, cy2):
                        conf = min(0.88, 0.62 + ea_ratio * 1.8)
                        if conf > confidence and conf >= 0.65:
                            confidence, device_type, detected = conf, 'bluetooth_earpiece', True

        should_log_device = False
        if session_id:
            sess_d = db.session.get(InterviewSession, session_id)
            if sess_d and sess_d.status == 'active' and sess_d.started_at:
                if (datetime.utcnow() - sess_d.started_at).total_seconds() >= 10:
                    should_log_device = True

        if detected and should_log_device:
            _, buf = cv2.imencode('.jpg', cv2.resize(frame, (160, 120)))
            thumb  = base64.b64encode(buf).decode('utf-8')
            da = DeviceAlert(user_id=current_user.id, session_id=session_id,
                             device_type=device_type, confidence=confidence, image_b64=thumb)
            db.session.add(da)
            db.session.commit()
            log_violation_db(current_user.id, session_id, 'device_detected',
                             device_data={'device_type': device_type, 'confidence': round(confidence, 2)})

        return jsonify({'success': True, 'phone_detected': detected,
                        'device_type': device_type, 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@proctoring_bp.route('/log-violation', methods=['POST'])
@login_required
def log_violation_route():
    data = request.get_json() or {}
    session_id = data.get('session_id')
    vtype = data.get('violation_type')

    if session_id:
        sess = db.session.get(InterviewSession, session_id)
        if not sess or sess.status != 'active' or not sess.started_at:
            return jsonify({'success': True, 'skipped': 'session not active yet'})
        elapsed = (datetime.utcnow() - sess.started_at).total_seconds()
        if elapsed < 5:
            return jsonify({'success': True, 'skipped': 'grace period'})

    log_violation_db(current_user.id, session_id, vtype)
    return jsonify({'success': True})


@proctoring_bp.route('/get-credibility-score')
@login_required
def get_credibility():
    session_id = request.args.get('session_id', type=int)
    score   = calc_credibility(session_id) if session_id else 100
    v_count = Violation.query.filter_by(session_id=session_id).count() if session_id else 0
    return jsonify({'success': True, 'credibility_score': score, 'total_violations': v_count})
