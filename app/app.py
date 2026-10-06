"""
RetinaEdge — diabetic retinopathy grading on the Raspberry Pi 5.

Run the web app:     python app.py                    (http://<pi-ip>:5000)
Benchmark models:    python app.py --benchmark        (writes benchmark_results.csv/json)

Threads: one captures camera frames, one classifies the latest frame while live mode is on,
Flask serves the page. The interpreter is shared, so every inference goes through ENGINE_LOCK.
"""
import argparse
import base64
import csv
import io
import json
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image, UnidentifiedImageError

from engine import Engine

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
PROC = psutil.Process()


# ---------------------------------------------------------------- system metrics
def cpu_temperature():
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def system_metrics():
    return {"cpu_percent": psutil.cpu_percent(interval=None),
            "ram_mb": round(PROC.memory_info().rss / 1e6, 1),
            "ram_percent": psutil.virtual_memory().percent,
            "temp_c": cpu_temperature()}


# ---------------------------------------------------------------- camera
class Camera:
    """Picamera2 (CSI camera) first, then a USB webcam via OpenCV. available=False if neither works."""

    def __init__(self, width=1280, height=960):
        self.frame, self.lock, self.available, self.backend, self.error = None, threading.Lock(), False, None, None
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            # Picamera2's "BGR888" is stored as R,G,B in memory, i.e. an RGB numpy array (naming quirk).
            self.cam.configure(self.cam.create_preview_configuration(main={"size": (width, height), "format": "BGR888"}))
            self.cam.start()
            self._read = lambda: self.cam.capture_array()
            self.backend = "picamera2"
        except Exception as e_pi:  # noqa: BLE001 - any failure means "try the next backend"
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("no USB camera at index 0")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                def read():
                    ok, f = cap.read()
                    if not ok:
                        raise RuntimeError("USB camera read failed")
                    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                self._read, self.backend = read, "opencv"
            except Exception as e_cv:  # noqa: BLE001
                self.error = f"no camera (picamera2: {e_pi}; opencv: {e_cv})"
                return
        self.available = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                f = self._read()
                with self.lock:
                    self.frame = f
            except Exception as e:  # noqa: BLE001 - keep the app alive, show the error on the page
                self.error = str(e)
                time.sleep(0.5)
            time.sleep(0.03)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ---------------------------------------------------------------- app state
app = Flask(__name__)
ENGINE: Engine = None
ENGINE_LOCK = threading.Lock()
CAMERA: Camera = None
STATE = {"live": False, "last": None, "fps": 0.0, "preview_jpeg": None, "error": None}


def to_jpeg(arr_or_pil, quality=80):
    im = arr_or_pil if isinstance(arr_or_pil, Image.Image) else Image.fromarray(arr_or_pil)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def run_classify(image):
    """Classifies one PIL image under the engine lock; strips the preview image from the JSON."""
    with ENGINE_LOCK:
        res = ENGINE.classify(image)
    preview = res.pop("preview")
    return res, to_jpeg(preview)


def live_loop():
    """While live mode is on, classify the newest camera frame back-to-back and track FPS."""
    times = []
    while True:
        if not (STATE["live"] and CAMERA and CAMERA.available):
            times.clear()
            time.sleep(0.1)
            continue
        frame = CAMERA.latest()
        if frame is None:
            time.sleep(0.05)
            continue
        t0 = time.perf_counter()
        try:
            res, jpeg = run_classify(Image.fromarray(frame))
            STATE.update(last=res, preview_jpeg=jpeg, error=None)
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the loop
            STATE["error"] = f"inference failed: {e}"
        times.append(time.perf_counter() - t0)
        times[:] = times[-10:]
        STATE["fps"] = round(len(times) / sum(times), 2)


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = CAMERA.latest() if CAMERA and CAMERA.available else None
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + to_jpeg(frame, 70) + b"\r\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/preview.jpg")
def preview():
    if STATE["preview_jpeg"] is None:
        return Response(status=204)
    return Response(STATE["preview_jpeg"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/status")
def status():
    return jsonify(live=STATE["live"], fps=STATE["fps"], last=STATE["last"], error=STATE["error"] or (CAMERA.error if CAMERA else None),
                   camera=CAMERA.backend if CAMERA and CAMERA.available else None,
                   model=ENGINE.model_key, model_mb=round(ENGINE.size_mb, 2), system=system_metrics())


@app.route("/api/models")
def models():
    return jsonify(current=ENGINE.model_key,
                   models={k: {"variant": v.get("variant"), "backbone": v.get("backbone"),
                               "test": v.get("test")} for k, v in ENGINE.models.items()})


@app.route("/api/model", methods=["POST"])
def set_model():
    key = (request.get_json(silent=True) or {}).get("key")
    try:
        with ENGINE_LOCK:
            ENGINE.load(key)
    except (KeyError, FileNotFoundError, ValueError) as e:
        return jsonify(error=str(e)), 400
    STATE["last"] = None
    return jsonify(model=ENGINE.model_key)


@app.route("/api/live", methods=["POST"])
def set_live():
    on = bool((request.get_json(silent=True) or {}).get("on"))
    if on and not (CAMERA and CAMERA.available):
        return jsonify(error=CAMERA.error if CAMERA else "camera disabled"), 409
    STATE["live"] = on
    return jsonify(live=on)


def _respond(image):
    res, jpeg = run_classify(image)
    STATE.update(last=res, preview_jpeg=jpeg)
    return jsonify(result=res, preview="data:image/jpeg;base64," + base64.b64encode(jpeg).decode())


@app.route("/api/classify", methods=["POST"])
def classify_upload():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="no file sent"), 400
    try:
        image = Image.open(f.stream)
        image.load()
    except (UnidentifiedImageError, OSError):
        return jsonify(error="not a readable image file"), 400
    return _respond(image)


@app.route("/api/samples")
def samples():
    return jsonify(sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}))


@app.route("/api/sample/<name>", methods=["POST"])
def classify_sample(name):
    path = (SAMPLES / name).resolve()
    if path.parent != SAMPLES.resolve() or not path.is_file():
        return jsonify(error="unknown sample"), 404
    with Image.open(path) as im:
        im.load()
        return _respond(im)


# ---------------------------------------------------------------- benchmark
def true_grade(name):
    """Samples are named with their grade, e.g. 'grade2_abc123.png'."""
    digits = [c for c in name.split("_")[0] if c.isdigit()]
    return int(digits[0]) if digits else None


def benchmark(engine, runs, out_dir):
    images = [(p.name, Image.open(p).convert("RGB")) for p in sorted(SAMPLES.glob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not images:
        raise SystemExit(f"no images in {SAMPLES}")
    rows = []
    for key in engine.models:
        rss_before = PROC.memory_info().rss
        engine.load(key)
        rss_model = (PROC.memory_info().rss - rss_before) / 1e6
        engine.classify(images[0][1])                    # warm-up: first inference is always slower, discarded
        psutil.cpu_percent(interval=None)
        inf, pre, total, correct, n_ok = [], [], [], 0, 0
        t_start = time.perf_counter()
        for _ in range(runs):
            for name, im in images:
                t0 = time.perf_counter()
                res = engine.classify(im)
                total.append((time.perf_counter() - t0) * 1000)
                pre.append(res["preprocess_ms"])
                if "inference_ms" in res:
                    inf.append(res["inference_ms"])
                    n_ok += 1
                    correct += res["grade"] == true_grade(name)
        elapsed = time.perf_counter() - t_start
        q = statistics.quantiles(inf, n=20) if len(inf) >= 2 else [float("nan")] * 19
        rows.append({"model": key, "variant": engine.spec.get("variant"), "size_mb": round(engine.size_mb, 2),
                     "inference_ms_mean": round(statistics.mean(inf), 1), "inference_ms_p50": round(statistics.median(inf), 1),
                     "inference_ms_p95": round(q[18], 1), "preprocess_ms_mean": round(statistics.mean(pre), 1),
                     "end_to_end_ms_mean": round(statistics.mean(total), 1),
                     "fps_end_to_end": round(len(total) / elapsed, 2),
                     "cpu_percent": psutil.cpu_percent(interval=None), "model_ram_mb": round(rss_model, 1),
                     "temp_c": cpu_temperature(), "sample_accuracy": round(correct / max(n_ok, 1), 3),
                     "n_inferences": len(inf)})
        print(json.dumps(rows[-1]))
    out = Path(out_dir)
    with open(out / "benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (out / "benchmark_results.json").write_text(json.dumps(rows, indent=2))
    print(f"saved {out / 'benchmark_results.csv'}")
    return rows


# ---------------------------------------------------------------- page
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>RetinaEdge</title>
<style>
:root{--bg:#0d0b0a;--panel:#171311;--line:#2b2420;--ink:#efe6dc;--dim:#9b8d80;--disc:#e98b3a;--ok:#7fb069;--warn:#e3b23c;--bad:#d1495b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "DejaVu Sans","Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:baseline;gap:16px;padding:18px 24px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:22px;letter-spacing:.02em;font-weight:600}h1 span{color:var(--disc)}
header p{margin:0;color:var(--dim);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:20px;padding:20px 24px;max-width:1300px}
@media(max-width:860px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
h2{margin:0 0 12px;font-size:14px;color:var(--dim);font-weight:500}
.feeds{display:grid;grid-template-columns:2fr 1fr;gap:12px}.feeds img{width:100%;border-radius:6px;background:#000;aspect-ratio:4/3;object-fit:contain}
.feeds figure{margin:0}.feeds figcaption{color:var(--dim);font-size:12px;margin-top:4px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;align-items:center}
button,select,label.btn{background:#231c18;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer}
button.primary{background:var(--disc);color:#1a0f06;border-color:var(--disc);font-weight:600}button:disabled{opacity:.4;cursor:default}
input[type=file]{display:none}
.grade{font-size:44px;font-weight:600;line-height:1.1}.gname{font-size:18px;color:var(--dim)}
.ref{display:inline-block;margin:10px 0;padding:4px 10px;border-radius:999px;font-weight:600;font-size:13px}
.ref.yes{background:var(--bad);color:#fff}.ref.no{background:var(--ok);color:#10200a}
.msg{margin:10px 0;padding:10px;border-left:3px solid var(--warn);background:#221c12;color:var(--warn);font-size:14px}
.bars{display:grid;grid-template-columns:auto 1fr 3.5em;gap:4px 8px;align-items:center;font-size:13px;margin-top:8px}
.bar{height:8px;background:#2a221d;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--disc)}
.num{font-family:"DejaVu Sans Mono",ui-monospace,monospace;text-align:right}
dl{display:grid;grid-template-columns:1fr auto;gap:4px 12px;margin:0;font-size:14px}dt{color:var(--dim)}dd{margin:0;font-family:"DejaVu Sans Mono",ui-monospace,monospace}
.samples{display:flex;flex-wrap:wrap;gap:6px}.samples button{padding:4px 8px;font-size:12px}
footer{padding:0 24px 24px;color:var(--dim);font-size:12px;max-width:1300px}
</style></head><body>
<header><h1>Retina<span>Edge</span></h1><p>Diabetic retinopathy grading (ICDR 0–4) on Raspberry Pi 5 · research prototype, not a medical device</p></header>
<main>
 <div>
  <section>
   <h2>Camera (recapture of a fundus photo on screen)</h2>
   <div class="feeds">
    <figure><img id="feed" src="/video_feed" alt="camera feed"><figcaption id="cam">camera: …</figcaption></figure>
    <figure><img id="prev" alt="what the model sees"><figcaption>What the model sees (cropped, padded, resized)</figcaption></figure>
   </div>
   <div class="controls">
    <button class="primary" id="live">Start live</button>
    <label class="btn">Upload image<input type="file" id="file" accept="image/*"></label>
    <select id="model"></select>
   </div>
  </section>
  <section style="margin-top:20px"><h2>Bundled test samples (true grade in the file name)</h2><div class="samples" id="samples"></div></section>
 </div>
 <div>
  <section id="result"><h2>Result</h2><p style="color:var(--dim)">Start live mode, upload an image or pick a sample.</p></section>
  <section style="margin-top:20px"><h2>Device</h2><dl id="sys"></dl></section>
 </div>
</main>
<footer>Grade from the expected grade cut by thresholds fitted on validation; referable = grade ≥ 2 (P(2)+P(3)+P(4) above the validation cut-off).</footer>
<script>
const $=id=>document.getElementById(id);const NAMES=["No DR","Mild","Moderate","Severe","Proliferative DR"];
function show(r){if(!r)return;let h='<h2>Result</h2>';
 if(r.status==="ok"||r.status==="low_confidence"){h+=`<div class="grade">Grade ${r.grade}</div><div class="gname">${r.grade_name}</div>
  <span class="ref ${r.referable?'yes':'no'}">${r.referable?'Refer to ophthalmologist':'Not referable'} · P=${r.p_referable.toFixed(2)}</span>`;
  if(r.message)h+=`<div class="msg">${r.message}</div>`;
  h+='<div class="bars">'+r.probs.map((p,i)=>`<span>${i} ${NAMES[i]}</span><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div><span class="num">${(p*100).toFixed(1)}%</span>`).join('')+'</div>';
  h+=`<dl style="margin-top:12px"><dt>Inference</dt><dd>${r.inference_ms} ms</dd><dt>Preprocess</dt><dd>${r.preprocess_ms} ms</dd><dt>Model</dt><dd>${r.model}</dd></dl>`;
 }else{h+=`<div class="msg">${r.message}</div>`}
 $('result').innerHTML=h;}
async function poll(){try{const s=await (await fetch('/api/status')).json();
 $('live').textContent=s.live?'Stop live':'Start live';$('cam').textContent='camera: '+(s.camera||'unavailable')+(s.error?' · '+s.error:'');
 if(s.live&&s.last){show(s.last);$('prev').src='/preview.jpg?'+Date.now();}
 const m=s.system;$('sys').innerHTML=`<dt>Inference FPS (live)</dt><dd>${s.live?s.fps:'–'}</dd><dt>CPU</dt><dd>${m.cpu_percent}%</dd>
 <dt>RAM (process)</dt><dd>${m.ram_mb} MB</dd><dt>RAM (system)</dt><dd>${m.ram_percent}%</dd><dt>Temperature</dt><dd>${m.temp_c??'–'} °C</dd>
 <dt>Model file</dt><dd>${s.model_mb} MB</dd>`;}catch(e){}}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return [r.ok,await r.json()];}
function done([ok,j]){if(!ok){$('result').innerHTML=`<h2>Result</h2><div class="msg">${j.error}</div>`;return;}show(j.result);$('prev').src=j.preview;}
$('live').onclick=async()=>{const on=$('live').textContent.startsWith('Start');const [ok,j]=await post('/api/live',{on});if(!ok)done([ok,j]);poll();};
$('file').onchange=async e=>{const f=e.target.files[0];if(!f)return;const fd=new FormData();fd.append('image',f);
 const r=await fetch('/api/classify',{method:'POST',body:fd});done([r.ok,await r.json()]);e.target.value='';};
$('model').onchange=async e=>{const [ok,j]=await post('/api/model',{key:e.target.value});if(!ok)done([ok,j]);};
(async()=>{const m=await (await fetch('/api/models')).json();
 $('model').innerHTML=Object.entries(m.models).map(([k,v])=>`<option value="${k}" ${k===m.current?'selected':''}>${k}</option>`).join('');
 const s=await (await fetch('/api/samples')).json();
 $('samples').innerHTML=s.map(n=>`<button data-n="${n}">${n}</button>`).join('');
 $('samples').onclick=async e=>{const n=e.target.dataset.n;if(n)done(await post('/api/sample/'+encodeURIComponent(n)));};})();
setInterval(poll,1000);poll();
</script></body></html>"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="RetinaEdge app")
    ap.add_argument("--model", help="model key from config.json (default: config default_model)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-camera", action="store_true", help="upload/sample mode only")
    ap.add_argument("--benchmark", action="store_true", help="benchmark every model on samples/ and exit")
    ap.add_argument("--runs", type=int, default=5, help="passes over samples/ per model in benchmark mode")
    args = ap.parse_args()

    global ENGINE, CAMERA
    ENGINE = Engine(HERE, args.model, num_threads=args.threads)
    if args.benchmark:
        benchmark(ENGINE, args.runs, HERE)
        return
    CAMERA = None if args.no_camera else Camera()
    if CAMERA and not CAMERA.available:
        print("WARNING:", CAMERA.error, "- running in upload/sample mode")
    threading.Thread(target=live_loop, daemon=True).start()
    psutil.cpu_percent(interval=None)
    print(f"RetinaEdge on http://0.0.0.0:{args.port}  model={ENGINE.model_key}  threads={args.threads}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
