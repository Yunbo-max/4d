"""Loopback-only bridge to five pinned public Hugging Face snapshots.

The GPU runs the unmodified HF client. Hub metadata/small files travel through
the control machine; external CDN redirects remain direct GPU downloads.
No request credentials are forwarded, and no arbitrary destination is accepted.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.parse
import urllib.request

PINS = {
    'facebook/actionbench': '2796071cbe6248422fcbeab3101fa9f9886cb7b9',
    'facebook/ActionMesh': 'fb69228ba8a4df684907b5d259cff3c22fb722f1',
    'VAST-AI/TripoSG': '2c1c516d22d58db486a058d98d31bb6177344e06',
    'facebook/dinov2-large': '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c',
    'briaai/RMBG-1.4': '2ceba5a5efaec153162aedea169f76caf9b46cf8',
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def bridge(self):
        parsed=urllib.parse.urlsplit(self.path)
        path=urllib.parse.unquote(parsed.path)
        if parsed.scheme or parsed.netloc or not any('/'+repo+'/' in path and rev in path for repo,rev in PINS.items()):
            self.send_error(403, 'Pinned public repository path required');return
        request=urllib.request.Request('https://huggingface.co'+self.path,
            method=self.command, headers={'User-Agent':'4d-public-snapshot-staging','Accept-Encoding':'identity'})
        try:
            try: response=urllib.request.build_opener(NoRedirect).open(request,timeout=30)
            except urllib.error.HTTPError as error: response=error
            with response:
                body=b'' if self.command=='HEAD' else response.read(8*1024*1024+1)
                if len(body)>8*1024*1024:
                    self.send_error(502,'Large objects must use direct CDN');return
                self.send_response(response.status)
                for key,value in response.headers.items():
                    if key.lower() in ('connection','transfer-encoding','content-length'):continue
                    if key.lower() in ('location','link'):
                        value=value.replace('https://huggingface.co/',self.server.endpoint+'/')
                    self.send_header(key,value)
                length=response.headers.get('Content-Length','0') if self.command=='HEAD' else str(len(body))
                self.send_header('Content-Length',length);self.end_headers()
                if body:self.wfile.write(body)
        except (OSError,urllib.error.URLError):
            self.send_error(502,'Public upstream unavailable')

    do_GET=bridge
    do_HEAD=bridge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=18765)
    p.add_argument('--remote-port',type=int,default=18766)
    a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    server.endpoint='http://127.0.0.1:'+str(a.remote_port)
    print('Pinned public Hub bridge listening on loopback',a.port,flush=True)
    server.serve_forever()


if __name__=='__main__':main()
