"""
Age & Gender Detection Widget (v3)
===================================
Drag-and-drop or browse an image → get age/gender prediction with green
bounding box overlay and full JSON output.

Run:  python realtime_age_gender_v3.py
Open: http://localhost:5000
"""

import base64
import io
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response

# ── DeepFace bootstrap ──────────────────────────────────────────────────
AGE_GENDER_ACTIONS = ["age", "gender"]
AGE_OFFSET = 6  # subtract from detected age (calibration)


def _local_deepface_path() -> Path:
    return Path(__file__).resolve().parent / "deepface"


def _ensure_deepface() -> None:
    for stream in (sys.stdout, sys.stderr):
        fn = getattr(stream, "reconfigure", None)
        if fn:
            fn(encoding="utf-8")
    os.environ.setdefault("DEEPFACE_HOME", str(Path(__file__).resolve().parent))
    p = _local_deepface_path()
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# ── Analysis helpers ─────────────────────────────────────────────────────
def analyze_image(img_array: np.ndarray, detector: str = "opencv") -> List[Dict[str, Any]]:
    _ensure_deepface()
    from deepface import DeepFace

    raw = DeepFace.analyze(
        img_path=img_array,
        actions=AGE_GENDER_ACTIONS,
        detector_backend=detector,
        enforce_detection=False,
        silent=True,
    )
    faces: List[Dict[str, Any]] = []
    for i, f in enumerate(raw):
        gender = str(f.get("dominant_gender", "Unknown"))
        scores = f.get("gender") or {}
        conf = float(scores.get(gender, 0))
        region = f.get("region") or {}
        raw_age = int(round(float(f.get("age", 0))))
        faces.append({
            "face_index": i,
            "age": max(0, raw_age - AGE_OFFSET),
            "raw_age": raw_age,
            "gender": gender,
            "gender_confidence": round(conf, 4),
            "bbox": {
                "x": int(region.get("x", 0)),
                "y": int(region.get("y", 0)),
                "w": int(region.get("w", 0)),
                "h": int(region.get("h", 0)),
            },
            "face_confidence": round(float(f.get("face_confidence", 0)), 6),
        })
    return faces


def draw_overlay(img: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
    GREEN = (0, 255, 0)
    out = img.copy()
    for f in faces:
        b = f["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        label = f'{f["age"]}  {f["gender"]}  {f["gender_confidence"]:.1f}%'
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, 2)
        # Background bar for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x, max(0, y - th - 14)), (x + tw + 6, max(0, y - 2)), (0, 0, 0), -1)
        cv2.putText(out, label, (x + 3, max(th + 4, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    return out


def img_to_b64(img: np.ndarray, fmt: str = ".jpg") -> str:
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf).decode()


# ── Flask app ────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_PAGE, content_type="text/html; charset=utf-8")


@app.route("/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    faces = analyze_image(img)
    overlay = draw_overlay(img, faces)

    return jsonify({
        "timestamp": _timestamp(),
        "faces": faces,
        "total_faces": len(faces),
        "overlay_b64": img_to_b64(overlay),
        "original_b64": img_to_b64(img),
    })


# ── Embedded HTML / CSS / JS ────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Age &amp; Gender Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0c0f1a;--surface:#141829;--surface2:#1c2038;--border:#2a2f4a;
    --accent:#00e676;--accent2:#00c853;--text:#e8eaf6;--text2:#9ea4c1;
    --danger:#ff5252;--radius:14px;--shadow:0 8px 32px rgba(0,0,0,.45);
  }
  html{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  body{display:flex;flex-direction:column;align-items:center;padding:32px 16px 64px;min-height:100vh}

  /* ── Header ─────────────────────────── */
  .header{text-align:center;margin-bottom:36px}
  .header h1{font-size:2rem;font-weight:800;letter-spacing:-.5px;
    background:linear-gradient(135deg,var(--accent),#69f0ae,#b9f6ca);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header p{color:var(--text2);margin-top:6px;font-size:.95rem}

  /* ── Layout ─────────────────────────── */
  .container{display:grid;grid-template-columns:1fr 1fr;gap:28px;width:100%;max-width:1100px}
  @media(max-width:760px){.container{grid-template-columns:1fr}}

  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:24px;box-shadow:var(--shadow);transition:border-color .3s}
  .card:hover{border-color:var(--accent)}
  .card-title{font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;
    color:var(--accent);margin-bottom:16px;display:flex;align-items:center;gap:8px}
  .card-title .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 8px var(--accent)}

  /* ── Drop zone ──────────────────────── */
  .dropzone{border:2px dashed var(--border);border-radius:12px;padding:48px 24px;
    text-align:center;cursor:pointer;transition:all .3s;position:relative;overflow:hidden;
    background:var(--surface2)}
  .dropzone.over{border-color:var(--accent);background:rgba(0,230,118,.06)}
  .dropzone.has-image{padding:0;border-style:solid;border-color:var(--accent)}
  .dropzone img{width:100%;border-radius:10px;display:block}
  .dropzone-text{color:var(--text2);font-size:.95rem;pointer-events:none}
  .dropzone-text .icon{font-size:2.4rem;margin-bottom:10px;display:block;opacity:.5}
  .dropzone-text strong{color:var(--text);font-weight:600}
  .dropzone input[type=file]{display:none}

  .btn{display:inline-flex;align-items:center;gap:8px;padding:12px 28px;border:none;
    border-radius:10px;font-family:inherit;font-size:.9rem;font-weight:600;cursor:pointer;
    transition:all .25s}
  .btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0c0f1a;
    box-shadow:0 4px 20px rgba(0,230,118,.25)}
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 6px 28px rgba(0,230,118,.35)}
  .btn-primary:disabled{opacity:.4;pointer-events:none;transform:none}
  .btn-secondary{background:var(--surface2);color:var(--text);border:1px solid var(--border)}
  .btn-secondary:hover{border-color:var(--accent);color:var(--accent)}

  .actions{display:flex;gap:12px;margin-top:18px;flex-wrap:wrap}

  /* ── Result image ───────────────────── */
  .result-img-wrap{position:relative;border-radius:12px;overflow:hidden;background:var(--surface2);
    min-height:200px;display:flex;align-items:center;justify-content:center}
  .result-img-wrap img{width:100%;display:block;border-radius:12px}
  .result-img-wrap .placeholder{color:var(--text2);font-size:.85rem;padding:40px;text-align:center}

  /* ── Toggle ─────────────────────────── */
  .toggle-bar{display:flex;gap:6px;margin-bottom:14px;background:var(--surface2);
    border-radius:10px;padding:4px;width:fit-content}
  .toggle-bar button{padding:8px 18px;border:none;border-radius:8px;font-family:inherit;
    font-size:.8rem;font-weight:600;cursor:pointer;background:transparent;color:var(--text2);
    transition:all .25s}
  .toggle-bar button.active{background:var(--accent);color:#0c0f1a}

  /* ── JSON panel ─────────────────────── */
  .json-wrap{background:#0a0d16;border:1px solid var(--border);border-radius:10px;
    padding:16px;max-height:400px;overflow-y:auto;font-size:.78rem;line-height:1.6}
  .json-wrap pre{white-space:pre-wrap;word-break:break-all;color:#b0bec5;margin:0}
  .json-wrap .key{color:#80cbc4}.json-wrap .str{color:#c5e1a5}
  .json-wrap .num{color:#ffcc80}.json-wrap .bool{color:#ef9a9a}

  /* ── Face cards ─────────────────────── */
  .face-cards{display:flex;flex-direction:column;gap:12px}
  .face-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
    padding:16px;display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:center}
  .face-card .avatar{width:56px;height:56px;border-radius:50%;border:2px solid var(--accent);
    display:flex;align-items:center;justify-content:center;font-size:1.5rem;background:rgba(0,230,118,.08)}
  .face-card .info{display:flex;flex-direction:column;gap:4px}
  .face-card .age-gender{font-size:1.15rem;font-weight:700}
  .face-card .age-gender .age{color:var(--accent)}
  .face-card .meta{font-size:.78rem;color:var(--text2)}
  .face-card .conf-bar{height:4px;border-radius:2px;background:var(--border);margin-top:4px;overflow:hidden}
  .face-card .conf-bar .fill{height:100%;border-radius:2px;background:var(--accent);transition:width .6s}

  /* ── Spinner ────────────────────────── */
  .spinner{display:none;align-items:center;gap:10px;padding:20px 0;color:var(--text2);font-size:.85rem}
  .spinner.show{display:flex}
  .spinner .ring{width:22px;height:22px;border:3px solid var(--border);border-top-color:var(--accent);
    border-radius:50%;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* ── Empty state ────────────────────── */
  .empty{text-align:center;padding:40px 20px;color:var(--text2);font-size:.85rem}
  .empty .icon{font-size:2rem;margin-bottom:8px;display:block;opacity:.4}

  /* ── Scrollbar ──────────────────────── */
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="header">
  <h1>&#x1F9E0; Age &amp; Gender Detector</h1>
  <p>Drag &amp; drop an image or browse — get instant AI predictions</p>
</div>

<div class="container">
  <!-- LEFT COLUMN -->
  <div style="display:flex;flex-direction:column;gap:24px">
    <!-- Upload card -->
    <div class="card">
      <div class="card-title"><span class="dot"></span>Upload Image</div>
      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <div class="dropzone-text" id="dropText">
          <span class="icon">&#128247;</span>
          <strong>Drop image here</strong><br>or click to browse<br>
          <span style="font-size:.78rem;margin-top:8px;display:inline-block">JPG, PNG, WEBP</span>
        </div>
        <input type="file" id="fileInput" accept="image/*">
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="analyzeBtn" disabled>&#x1F50D; Analyze</button>
        <button class="btn btn-secondary" id="clearBtn" style="display:none">&#x2715; Clear</button>
      </div>
      <div class="spinner" id="spinner"><div class="ring"></div>Analyzing faces…</div>
    </div>

    <!-- Detected faces -->
    <div class="card">
      <div class="card-title"><span class="dot"></span>Detected Faces</div>
      <div id="faceCards">
        <div class="empty"><span class="icon">&#128100;</span>No faces detected yet</div>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN -->
  <div style="display:flex;flex-direction:column;gap:24px">
    <!-- Result image -->
    <div class="card">
      <div class="card-title"><span class="dot"></span>Result</div>
      <div class="toggle-bar">
        <button class="active" id="togOverlay" onclick="showView('overlay')">Overlay</button>
        <button id="togOriginal" onclick="showView('original')">Original</button>
      </div>
      <div class="result-img-wrap" id="resultWrap">
        <div class="placeholder">Result will appear here</div>
      </div>
    </div>

    <!-- JSON output -->
    <div class="card">
      <div class="card-title"><span class="dot"></span>JSON Output</div>
      <div class="json-wrap" id="jsonWrap">
        <pre id="jsonPre">{ "status": "waiting for image" }</pre>
      </div>
    </div>
  </div>
</div>

<script>
const dropzone   = document.getElementById('dropzone');
const dropText   = document.getElementById('dropText');
const fileInput  = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn   = document.getElementById('clearBtn');
const spinner    = document.getElementById('spinner');
const resultWrap = document.getElementById('resultWrap');
const jsonPre    = document.getElementById('jsonPre');
const faceCards  = document.getElementById('faceCards');

let selectedFile = null;
let overlayB64   = null;
let originalB64  = null;
let currentView  = 'overlay';

/* ── Drag & Drop ───────────────────── */
['dragenter','dragover'].forEach(e=>{
  dropzone.addEventListener(e,ev=>{ev.preventDefault();dropzone.classList.add('over')})
});
['dragleave','drop'].forEach(e=>{
  dropzone.addEventListener(e,ev=>{ev.preventDefault();dropzone.classList.remove('over')})
});
dropzone.addEventListener('drop',ev=>{
  const f=ev.dataTransfer.files[0];
  if(f && f.type.startsWith('image/')) loadFile(f);
});
fileInput.addEventListener('change',()=>{if(fileInput.files[0]) loadFile(fileInput.files[0])});

function loadFile(f){
  selectedFile=f;
  const reader=new FileReader();
  reader.onload=e=>{
    dropzone.classList.add('has-image');
    dropText.style.display='none';
    let img=dropzone.querySelector('img');
    if(!img){img=document.createElement('img');dropzone.appendChild(img)}
    img.src=e.target.result;
    analyzeBtn.disabled=false;
    clearBtn.style.display='inline-flex';
  };
  reader.readAsDataURL(f);
}

clearBtn.addEventListener('click',()=>{
  selectedFile=null;overlayB64=null;originalB64=null;
  dropzone.classList.remove('has-image');
  dropText.style.display='';
  const img=dropzone.querySelector('img');if(img)img.remove();
  analyzeBtn.disabled=true;clearBtn.style.display='none';
  resultWrap.innerHTML='<div class="placeholder">Result will appear here</div>';
  jsonPre.textContent='{ "status": "waiting for image" }';
  faceCards.innerHTML='<div class="empty"><span class="icon">&#128100;</span>No faces detected yet</div>';
});

/* ── Analyze ───────────────────────── */
analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true;
  spinner.classList.add('show');
  resultWrap.innerHTML='<div class="placeholder">Processing…</div>';

  const fd=new FormData();
  fd.append('image',selectedFile);

  try{
    const res=await fetch('/analyze',{method:'POST',body:fd});
    const data=await res.json();
    if(data.error){throw new Error(data.error)}

    overlayB64=data.overlay_b64;
    originalB64=data.original_b64;
    showView(currentView);

    // JSON
    const display={timestamp:data.timestamp,total_faces:data.total_faces,faces:data.faces};
    jsonPre.innerHTML=syntaxHL(JSON.stringify(display,null,2));

    // Face cards
    if(data.faces.length===0){
      faceCards.innerHTML='<div class="empty"><span class="icon">&#128100;</span>No faces detected</div>';
    } else {
      faceCards.innerHTML=data.faces.map(f=>`
        <div class="face-card">
          <div class="avatar">&#128100;</div>
          <div class="info">
            <div class="age-gender">
              <span class="age">${f.age}</span> · ${f.gender}
            </div>
            <div class="meta">Confidence ${f.gender_confidence.toFixed(1)}% · Detection ${(f.face_confidence*100).toFixed(0)}%</div>
            <div class="conf-bar"><div class="fill" style="width:${f.gender_confidence}%"></div></div>
          </div>
        </div>
      `).join('');
    }
  } catch(err){
    resultWrap.innerHTML=`<div class="placeholder" style="color:var(--danger)">Error: ${err.message}</div>`;
  } finally {
    spinner.classList.remove('show');
    analyzeBtn.disabled=false;
  }
});

/* ── View toggle ───────────────────── */
function showView(v){
  currentView=v;
  document.getElementById('togOverlay').classList.toggle('active',v==='overlay');
  document.getElementById('togOriginal').classList.toggle('active',v==='original');
  const src=v==='overlay'?overlayB64:originalB64;
  if(src) resultWrap.innerHTML=`<img src="data:image/jpeg;base64,${src}">`;
}

/* ── JSON syntax highlight ─────────── */
function syntaxHL(s){
  return s.replace(/("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,m=>{
    let c='num';
    if(/^"/.test(m)){c=/:$/.test(m)?'key':'str'}
    else if(/true|false/.test(m))c='bool';
    return`<span class="${c}">${m}</span>`;
  });
}
</script>
</body>
</html>
"""

# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  [*] Age & Gender Detector v3")
    print("  -----------------------------")
    print("  Open  http://localhost:5000  in your browser")
    print("  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
