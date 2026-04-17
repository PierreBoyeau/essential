"""
Module viewer — serves metabolic map (SVG) + UMAP (PNG) side by side.
Run:  python viewer.py
Then open http://localhost:8765 in a browser.
Navigate with  ←  /  →  arrow keys, or the small button.
"""

import http.server
import json
import os
import re
import socketserver
import webbrowser
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
PORT = 8765


def discover_modules() -> list[str]:
    """Return sorted list of module IDs that have both SVG and PNG."""
    svgs = {p.stem for p in FIGURES_DIR.glob("*.svg")}
    pngs = {p.stem.removesuffix("_umap") for p in FIGURES_DIR.glob("*_umap.png")}
    both = svgs & pngs
    # Extract numeric part for sorting (e.g. eco_eco_M00001 -> 1)
    def sort_key(name: str) -> int:
        digits = re.sub(r"\D", "", name)
        return int(digits) if digits else 0
    return sorted(both, key=sort_key)


MODULES = discover_modules()

HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Module Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #111;
    color: #ddd;
    font-family: "SF Pro Text", system-ui, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  #header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 12px;
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
    flex-shrink: 0;
  }
  #module-label {
    font-size: 13px;
    letter-spacing: 0.04em;
    color: #aaa;
    flex: 1;
  }
  #counter {
    font-size: 11px;
    color: #555;
  }
  #btn-next {
    background: none;
    border: 1px solid #333;
    color: #777;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }
  #btn-next:hover { border-color: #666; color: #ccc; }
  #panels {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 6px;
  }
  .panel + .panel { border-top: 1px solid #1e1e1e; }
  .panel img, .panel object {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
  #no-umap {
    color: #333;
    font-size: 12px;
    display: none;
  }
</style>
</head>
<body>
<div id="header">
  <span id="module-label">—</span>
  <span id="counter"></span>
  <button id="btn-next" title="Next (→)">next →</button>
</div>
<div id="panels">
  <div class="panel" id="top-panel">
    <object id="svg-view" type="image/svg+xml" data=""></object>
  </div>
  <div class="panel" id="bot-panel">
    <img id="png-view" src="" alt="umap"/>
    <span id="no-umap">no UMAP</span>
  </div>
</div>

<script id="modules-data" type="application/json">MODULE_LIST_PLACEHOLDER</script>
<script>
const MODULES = JSON.parse(document.getElementById('modules-data').textContent);
let idx = 0;

function load(i) {
  idx = ((i % MODULES.length) + MODULES.length) % MODULES.length;
  const m = MODULES[idx];
  document.getElementById('module-label').textContent = m;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + MODULES.length;

  // Swap object element to force reload (setting .data alone doesn't always trigger reload)
  const top = document.getElementById('top-panel');
  const old = document.getElementById('svg-view');
  const obj = document.createElement('object');
  obj.id = 'svg-view';
  obj.type = 'image/svg+xml';
  obj.style.maxWidth = '100%';
  obj.style.maxHeight = '100%';
  obj.data = '/figures/' + m + '.svg?t=' + Date.now();
  top.replaceChild(obj, old);

  const png = document.getElementById('png-view');
  const noUmap = document.getElementById('no-umap');
  png.onerror = () => { png.style.display = 'none'; noUmap.style.display = ''; };
  png.onload  = () => { png.style.display = ''; noUmap.style.display = 'none'; };
  noUmap.style.display = 'none';
  png.src = '/figures/' + m + '_umap.png?t=' + Date.now();
}

document.getElementById('btn-next').addEventListener('click', () => load(idx + 1));

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') load(idx + 1);
  if (e.key === 'ArrowLeft')  load(idx - 1);
});

load(0);
</script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            page = HTML.replace("MODULE_LIST_PLACEHOLDER", json.dumps(MODULES))
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/figures/"):
            fname = self.path.split("?")[0].removeprefix("/figures/")
            fpath = FIGURES_DIR / fname
            if fpath.exists():
                data = fpath.read_bytes()
                ctype = "image/svg+xml" if fname.endswith(".svg") else "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass  # suppress request noise


if __name__ == "__main__":
    os.chdir(FIGURES_DIR.parent)
    print(f"Modules found: {len(MODULES)}")
    for m in MODULES:
        print(f"  {m}")
    print(f"\nServing at http://localhost:{PORT}")
    print("Navigate with  ←  /  →  arrow keys.\n")
    with socketserver.TCPServer(("", PORT), Handler) as srv:
        srv.allow_reuse_address = True
        webbrowser.open(f"http://localhost:{PORT}")
        srv.serve_forever()
