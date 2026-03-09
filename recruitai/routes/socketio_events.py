from flask import request
from flask_socketio import emit, join_room, leave_room
from extensions import socketio

_room_members = {}


def register_socketio_events(app):
    @socketio.on('connect')
    def on_connect():
        print(f'Connected: {request.sid}')

    @socketio.on('join_room_event')
    def on_join_room(data):
        room = data.get('room')
        role = data.get('role', 'candidate')
        join_room(room)

        if room not in _room_members:
            _room_members[room] = []
        _room_members[room] = [m for m in _room_members[room] if m['sid'] != request.sid]
        _room_members[room].append({'sid': request.sid, 'role': role})

        emit('user_joined', {'role': role, 'sid': request.sid}, room=room)

        if role == 'recruiter':
            emit('recruiter_joined', {'room': room}, room=room, include_self=False)
            existing = [m for m in _room_members[room] if m['role'] == 'candidate']
            if existing:
                emit('candidate_present', {'room': room, 'count': len(existing)})
        else:
            recruiters = [m for m in _room_members[room] if m['role'] == 'recruiter']
            if recruiters:
                emit('recruiter_joined', {'room': room})

    @socketio.on('leave_room_event')
    def on_leave_room(data):
        room = data.get('room')
        leave_room(room)
        if room in _room_members:
            _room_members[room] = [m for m in _room_members[room] if m['sid'] != request.sid]

    @socketio.on('candidate_ready')
    def on_candidate_ready(data):
        room = data.get('room')
        if room:
            emit('candidate_ready', data, room=room, include_self=False)

    @socketio.on('recruiter_joined_room')
    def on_recruiter_joined_room(data):
        room = data.get('room')
        if room:
            emit('recruiter_joined', {'room': room}, room=room, include_self=False)

    @socketio.on('webrtc_offer')
    def on_offer(data):
        emit('webrtc_offer', data, room=data['room'], include_self=False)

    @socketio.on('webrtc_answer')
    def on_answer(data):
        emit('webrtc_answer', data, room=data['room'], include_self=False)

    @socketio.on('webrtc_ice_candidate')
    def on_ice(data):
        emit('webrtc_ice_candidate', data, room=data['room'], include_self=False)

    @socketio.on('recruiter_message')
    def on_recruiter_msg(data):
        emit('recruiter_message', data, room=data['room'], include_self=False)

    @socketio.on('end_interview')
    def on_end_interview(data):
        emit('interview_ended', data, room=data['room'])

    @socketio.on('join_dashboard')
    def on_join_dashboard():
        join_room('dashboard')

    @socketio.on('disconnect')
    def on_disconnect():
        print(f'Disconnected: {request.sid}')
        for room in list(_room_members.keys()):
            leaving = [m for m in _room_members[room] if m['sid'] == request.sid]
            _room_members[room] = [m for m in _room_members[room] if m['sid'] != request.sid]
            for member in leaving:
                if member.get('role') == 'candidate':
                    emit('candidate_left', {'reason': 'disconnected', 'room': room}, room=room)
