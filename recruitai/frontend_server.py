"""
RecruitAI — Frontend + API Proxy Server
Serves HTML on port 3000 and proxies ALL non-static requests to Flask on port 5000.
Handles large file uploads (photos) and WebSocket upgrade (Socket.IO).

Terminal 1: python app.py              (Flask on :5000)
Terminal 2: python frontend_server.py  (Proxy+UI on :3000)
Browser:    http://localhost:3000
"""
import http.server
import socketserver
import os
import sys
import urllib.request
import urllib.error
import socket
import threading
import select

PORT      = 3000
API_PORT  = 5000
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

STATIC_EXT = ('.html', '.css', '.js', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg',
              '.woff', '.woff2', '.ttf', '.map')

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=TEMPLATES, **kwargs)

    def is_static(self):
        path = self.path.split('?')[0].split('#')[0]
        if path in ('/', ''):
            return True
        if any(path.lower().endswith(ext) for ext in STATIC_EXT):
            local = os.path.join(TEMPLATES, path.lstrip('/'))
            return os.path.isfile(local)
        return False

    def do_GET(self):
        # WebSocket upgrade — pipe directly
        if self.headers.get('Upgrade', '').lower() == 'websocket':
            self._tunnel()
            return
        if self.is_static():
            if self.path.split('?')[0] in ('/', ''):
                self.path = '/index.html'
            super().do_GET()
        else:
            self._proxy()

    def do_POST(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def _tunnel(self):
        """Raw TCP tunnel for WebSocket/Socket.IO — bypasses HTTP parsing."""
        try:
            backend = socket.create_connection(('localhost', API_PORT), timeout=10)
        except Exception as e:
            self.send_error(502, f'Tunnel failed: {e}')
            return

        # Forward the original request line + headers to backend
        req_line = f'{self.command} {self.path} HTTP/1.1\r\n'
        headers  = req_line.encode()
        for key, val in self.headers.items():
            if key.lower() == 'host':
                headers += f'Host: localhost:{API_PORT}\r\n'.encode()
            else:
                headers += f'{key}: {val}\r\n'.encode()
        headers += b'\r\n'
        backend.sendall(headers)

        client = self.connection
        client.setblocking(False)
        backend.setblocking(False)

        def pipe():
            while True:
                try:
                    r, _, _ = select.select([client, backend], [], [], 30)
                    if not r:
                        break
                    for s in r:
                        try:
                            data = s.recv(65536)
                            if not data:
                                return
                            (backend if s is client else client).sendall(data)
                        except Exception:
                            return
                except Exception:
                    break

        t = threading.Thread(target=pipe, daemon=True)
        t.start()
        t.join()
        try: backend.close()
        except: pass

    def _proxy(self):
        """HTTP proxy for API calls — handles large bodies (photos)."""
        target = f'http://localhost:{API_PORT}{self.path}'
        try:
            # Read full body — critical for large base64 photo uploads
            length = self.headers.get('Content-Length')
            if length:
                body = self.rfile.read(int(length))
            else:
                body = b''

            req = urllib.request.Request(target, data=body or None, method=self.command)

            # Forward all headers, especially Cookie for session auth
            for key, val in self.headers.items():
                low = key.lower()
                if low in ('host', 'connection', 'transfer-encoding'):
                    continue
                req.add_header(key, val)
            req.add_header('Host', f'localhost:{API_PORT}')
            if body:
                req.add_header('Content-Length', str(len(body)))

            with urllib.request.urlopen(req, timeout=60) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() in ('connection', 'transfer-encoding'):
                        continue
                    self.send_header(key, val)
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() in ('connection', 'transfer-encoding'):
                    continue
                self.send_header(key, val)
            self.end_headers()
            try: self.wfile.write(e.read())
            except: pass
        except Exception as e:
            print(f'  [ERR] {self.path}: {e}')
            try:
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"success":false,"message":"Server error: {e}"}}'.encode())
            except: pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def log_message(self, fmt, *args):
        kind = 'STATIC' if self.is_static() else 'PROXY '
        print(f'  [{kind}] {args[0]}')

class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads      = True

if __name__ == '__main__':
    os.chdir(TEMPLATES)
    print('=' * 55)
    print('  RecruitAI — Frontend + Proxy Server')
    print('=' * 55)
    print(f'  Open   → http://localhost:{PORT}')
    print(f'  API    → http://localhost:{API_PORT}  (proxied)')
    print(f'  WS     → ws://localhost:{PORT}        (tunnelled)')
    print('=' * 55)
    with ThreadedServer(('', PORT), ProxyHandler) as httpd:
        httpd.serve_forever()