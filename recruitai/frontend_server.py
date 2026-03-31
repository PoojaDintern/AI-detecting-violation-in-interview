"""
RecruitAI — Frontend Static Server + API Proxy
Serves HTML/CSS/JS on port 3000 AND proxies all API calls to Flask on port 5000.

THE KEY FIX: Python's http.client merges duplicate headers via email.message,
which silently drops extra Set-Cookie headers. We use get_all('Set-Cookie')
to forward every single cookie Flask sets.

Terminal 1: python app.py               (Flask backend — port 5000)
Terminal 2: python frontend_server.py   (UI + proxy   — port 3000)
Open:       http://localhost:3000
"""

import http.server
import socketserver
import urllib.request
import urllib.error
import http.client
import os

PORT = 3000
BACKEND = 'http://localhost:5000'
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

PROXY_PREFIXES = (
    '/api/',
    '/login',
    '/logout',
    '/signup',
    '/auth/',
)


class ProxyHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=TEMPLATES_DIR, **kwargs)

    def _should_proxy(self):
        p = self.path
        return any(p == prefix.rstrip('/') or p.startswith(prefix)
                   for prefix in PROXY_PREFIXES)

    def do_GET(self):
        if self._should_proxy():
            self._proxy()
        else:
            super().do_GET()

    def do_POST(self):
        if self._should_proxy():
            self._proxy()
        else:
            self.send_error(405)

    def do_PUT(self):    self._proxy()
    def do_DELETE(self): self._proxy()
    def do_PATCH(self):  self._proxy()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,PATCH,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization,Cookie')
        self.end_headers()

    def _proxy(self):
        # Parse host/path
        from urllib.parse import urlparse
        parsed = urlparse(BACKEND)
        host   = parsed.hostname
        port   = parsed.port or 80

        # Read body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build headers to forward
        fwd_headers = {}
        for key in ('Content-Type', 'Cookie', 'Authorization',
                    'Accept', 'X-Requested-With', 'User-Agent'):
            val = self.headers.get(key)
            if val:
                fwd_headers[key] = val
        fwd_headers['X-Forwarded-For'] = self.client_address[0]
        if body and 'Content-Type' not in fwd_headers:
            fwd_headers['Content-Type'] = 'application/json'

        try:
            # Use http.client directly so we get the raw HTTPResponse object
            # which has get_all() for duplicate headers like Set-Cookie
            conn = http.client.HTTPConnection(host, port, timeout=30)
            conn.request(self.command, self.path, body=body, headers=fwd_headers)
            resp = conn.getresponse()
            resp_body = resp.read()

            self.send_response(resp.status)

            # ── Forward headers — handle Set-Cookie specially ──────────────
            skip = {'transfer-encoding', 'connection', 'keep-alive'}
            forwarded_keys = set()

            for key, val in resp.getheaders():
                if key.lower() in skip:
                    continue
                if key.lower() == 'set-cookie':
                    continue  # handled separately below
                if key.lower() == 'location':
                    val = val.replace('localhost:5000', f'localhost:{PORT}')
                    val = val.replace('127.0.0.1:5000',  f'127.0.0.1:{PORT}')
                self.send_header(key, val)

            # Forward ALL Set-Cookie headers individually
            # getheaders() returns a list of (name, value) tuples
            # so multiple Set-Cookie entries are all preserved here
            for key, val in resp.getheaders():
                if key.lower() == 'set-cookie':
                    self.send_header('Set-Cookie', val)
                    print(f"  [COOKIE] Forwarding: {val[:80]}")

            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(resp_body)
            conn.close()

        except ConnectionRefusedError:
            self.send_error(502, "Backend not running — start app.py on port 5000")
        except Exception as ex:
            self.send_error(500, f"Proxy error: {ex}")

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        tag = '[PROXY ]' if self._should_proxy() else '[STATIC]'
        print(f"  {tag} {args[0]}")


if __name__ == '__main__':
    os.chdir(TEMPLATES_DIR)

    print("=" * 55)
    print("  RecruitAI — Frontend Server + API Proxy")
    print("=" * 55)
    print(f"  UI + Proxy:  http://localhost:{PORT}")
    print(f"  Backend API: http://localhost:5000  <- must be running")
    print("=" * 55)
    print("  Open http://localhost:3000 in your browser")
    print("  Cookies will be logged with [COOKIE] prefix")
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        httpd.serve_forever()