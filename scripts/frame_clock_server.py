#!/usr/bin/env python3
"""Serve one animated page and store its requestAnimationFrame intervals."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

out_path = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
port = int(sys.argv[3]) if len(sys.argv) > 3 else 8877

PAGE = """<!DOCTYPE html>
<meta charset="utf-8">
<title>GPUTronic frame clock</title>
<style>
  html, body { margin: 0; height: 100%%; background: #111; color: #eee;
    font: 20px sans-serif; overflow: hidden; }
  #box { width: 100px; height: 100px; background: #3cf; position: absolute; top: 160px; }
  #out { position: absolute; left: 24px; top: 24px; }
</style>
<body>
<div id="box"></div>
<pre id="out">waiting</pre>
<script>
const box = document.getElementById("box");
const out = document.getElementById("out");
const seconds = %s;
const samples = [];
function run() {
  let last = performance.now();
  const t0 = last;
  out.textContent = "running";
  function frame(now) {
    samples.push(now - last);
    last = now;
    box.style.left = (40 + ((now / 16) %% 900)) + "px";
    if (now - t0 < seconds * 1000) requestAnimationFrame(frame);
    else finish();
  }
  requestAnimationFrame(frame);
}
function finish() {
  out.textContent = "done " + samples.length;
  const body = JSON.stringify({
    w: window.innerWidth, h: window.innerHeight,
    sw: screen.width, sh: screen.height,
    dpr: window.devicePixelRatio,
    hidden: document.hidden,
    samples: samples
  });
  fetch("/frames", {method: "POST", headers: {"Content-Type": "application/json"}, body: body});
}
function arm() {
  if (document.hidden) {
    document.addEventListener("visibilitychange", function once() {
      if (!document.hidden) { document.removeEventListener("visibilitychange", once); setTimeout(run, 300); }
    });
  } else {
    setTimeout(run, 300);
  }
}
arm();
</script>
</body>
""" % seconds

# The page template uses %% so a literal percent survives the % format.
PAGE = PAGE.replace("%%", "%")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        data = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        self.send_response(204)
        self.end_headers()
        payload = json.loads(raw.decode())
        samples = payload.get("samples", [])
        with open(out_path, "w") as f:
            f.write("i,dt_ms,w,h,sw,sh,dpr,hidden\n")
            meta = (payload.get("w"), payload.get("h"), payload.get("sw"),
                    payload.get("sh"), payload.get("dpr"), payload.get("hidden"))
            for i, dt in enumerate(samples):
                f.write("%d,%.3f,%s,%s,%s,%s,%s,%s\n" % ((i, float(dt)) + meta))
        print(f"frames={len(samples)} hidden={payload.get('hidden')} "
              f"window={payload.get('w')}x{payload.get('h')} "
              f"screen={payload.get('sw')}x{payload.get('sh')} wrote={out_path}",
              flush=True)
        threading_stop()


def threading_stop():
    threading.Thread(target=server.shutdown, daemon=True).start()


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
print(f"listening {port}", flush=True)
server.serve_forever()
