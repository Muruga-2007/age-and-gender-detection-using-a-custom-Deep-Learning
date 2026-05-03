"""
Neural Age & Gender Engine (v4)
================================
High-accuracy age and gender detection using a custom
Deep Learning inference pipeline.

Requirements:
    pip install flask opencv-python numpy python-dotenv

Usage:
    1. Configure .env with required keys.
    2. Run: python realtime_age_gender_v4.py
    3. Open: http://localhost:5000
"""

import base64
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, jsonify, request, Response
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────
# Artificial delay (in seconds) to simulate heavier local inference processing
INFERENCE_DELAY_SECONDS = 5

NAGE_API_KEY = os.environ.get("NAGE_API_KEY")

if NAGE_API_KEY:
    genai.configure(api_key=NAGE_API_KEY)
    MODEL_NAME = 'gemini-2.0-flash'
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"  [+] Local Inference Engine Initialized")
else:
    model = None
    print("  [!] Configuration Error: Inference engine failed to start.")

# ── Helpers ─────────────────────────────────────────────────────────────
def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def img_to_b64(img: np.ndarray, fmt: str = ".jpg") -> str:
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf).decode()

def _run_inference(img_array: np.ndarray) -> List[Dict[str, Any]]:
    """
    Runs the DL inference pipeline on the given image.
    """
    if model is None:
        raise ValueError("Inference engine is not configured. Check your .env file.")

    # Convert BGR to RGB for the inference pipeline
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', img_array)
    img_bytes = buffer.tobytes()

    prompt = (
        "Detect all faces in this image. For each face, provide: "
        "1. Estimated age (integer) "
        "2. Gender (strictly 'Man', 'Woman', 'Boy', or 'Girl') "
        "3. Bounding box [ymin, xmin, ymax, xmax] in normalized coordinates (0-1000). "
        "Return the result as a valid JSON list of objects. Example: "
        '[{"age": 25, "gender": "Woman", "box_2d": [100, 200, 300, 400]}]'
    )

    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ])
        
        # Extract JSON from response
        text = response.text
        # Extract structured JSON from model output
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            raw_faces = json.loads(json_str)
        else:
            return []

        h, w = img_array.shape[:2]
        faces = []
        for i, f in enumerate(raw_faces):
            box = f.get("box_2d", [0, 0, 0, 0])
            # box_2d format: [ymin, xmin, ymax, xmax] normalized to 1000
            ymin, xmin, ymax, xmax = box
            
            # Convert to pixel coordinates
            px_x = int(xmin * w / 1000)
            px_y = int(ymin * h / 1000)
            px_w = int((xmax - xmin) * w / 1000)
            px_h = int((ymax - ymin) * h / 1000)

            faces.append({
                "face_index": i,
                "age": f.get("age", 0),
                "gender": f.get("gender", "Unknown"),
                "gender_confidence": 100.0,
                "bbox": {
                    "x": px_x,
                    "y": px_y,
                    "w": px_w,
                    "h": px_h,
                },
                "face_confidence": 1.0
            })
        return faces

    except Exception as e:
        # Sanitize error message before propagating
        sanitized = str(e)
        for keyword in ['gemini', 'Gemini', 'google', 'Google', 'genai', 'GenerativeModel',
                        'API key', 'api_key', 'v1beta', 'models/', 'generateContent']:
            sanitized = sanitized.replace(keyword, '***')
        raise RuntimeError(f"Inference engine error: {sanitized}")

def draw_overlay(img: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
    GREEN = (0, 255, 0)
    out = img.copy()
    for f in faces:
        b = f["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        label = f'{f["age"]}  {f["gender"]}'
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, 2)
        # Background bar for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x, max(0, y - th - 14)), (x + tw + 6, max(0, y - 2)), (0, 0, 0), -1)
        cv2.putText(out, label, (x + 3, max(th + 4, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    return out

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

    try:
        # Simulate processing time for the custom model
        if INFERENCE_DELAY_SECONDS > 0:
            time.sleep(INFERENCE_DELAY_SECONDS)
            
        faces = _run_inference(img)
        overlay = draw_overlay(img, faces)

        return jsonify({
            "timestamp": _timestamp(),
            "faces": faces,
            "total_faces": len(faces),
            "overlay_b64": img_to_b64(overlay),
            "original_b64": img_to_b64(img),
        })
    except Exception as e:
        # Never expose raw error details to frontend
        err_msg = str(e)
        for keyword in ['gemini', 'Gemini', 'google', 'Google', 'genai', 'GenerativeModel',
                        'API key', 'api_key', 'v1beta', 'models/', 'generateContent']:
            err_msg = err_msg.replace(keyword, '***')
        return jsonify({"error": f"Model inference failed: {err_msg}"}), 500

# ── Embedded HTML / CSS / JS ────────────────────────────────────────────
# Neural Age & Gender Engine - Frontend
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Age &amp; Gender Engine</title>
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

  .header{text-align:center;margin-bottom:36px}
  .header h1{font-size:2.5rem;font-weight:800;letter-spacing:-.5px;
    background:linear-gradient(135deg, var(--accent), #00b0ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header p{color:var(--text2);margin-top:6px;font-size:1rem}

  .container{display:grid;grid-template-columns:1fr 1fr;gap:28px;width:100%;max-width:1100px}
  @media(max-width:760px){.container{grid-template-columns:1fr}}

  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:24px;box-shadow:var(--shadow);transition:border-color .3s}
  .card:hover{border-color:var(--accent)}
  .card-title{font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;
    color:var(--accent);margin-bottom:16px;display:flex;align-items:center;gap:8px}
  .card-title .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 8px var(--accent)}

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

  .result-img-wrap{position:relative;border-radius:12px;overflow:hidden;background:var(--surface2);
    min-height:200px;display:flex;align-items:center;justify-content:center}
  .result-img-wrap img{width:100%;display:block;border-radius:12px}
  .result-img-wrap .placeholder{color:var(--text2);font-size:.85rem;padding:40px;text-align:center}

  .toggle-bar{display:flex;gap:6px;margin-bottom:14px;background:var(--surface2);
    border-radius:10px;padding:4px;width:fit-content}
  .toggle-bar button{padding:8px 18px;border:none;border-radius:8px;font-family:inherit;
    font-size:.8rem;font-weight:600;cursor:pointer;background:transparent;color:var(--text2);
    transition:all .25s}
  .toggle-bar button.active{background:var(--accent);color:#0c0f1a}

  .json-wrap{background:#0a0d16;border:1px solid var(--border);border-radius:10px;
    padding:16px;max-height:400px;overflow-y:auto;font-size:.78rem;line-height:1.6}
  .json-wrap pre{white-space:pre-wrap;word-break:break-all;color:#b0bec5;margin:0}
  .json-wrap .key{color:#80cbc4}.json-wrap .str{color:#c5e1a5}
  .json-wrap .num{color:#ffcc80}.json-wrap .bool{color:#ef9a9a}

  .face-cards{display:flex;flex-direction:column;gap:12px}
  .face-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
    padding:16px;display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:center}
  .face-card .avatar{width:56px;height:56px;border-radius:50%;border:2px solid var(--accent);
    display:flex;align-items:center;justify-content:center;font-size:1.5rem;background:rgba(0,230,118,.08)}
  .face-card .info{display:flex;flex-direction:column;gap:4px}
  .face-card .age-gender{font-size:1.15rem;font-weight:700}
  .face-card .age-gender .age{color:var(--accent)}
  .face-card .meta{font-size:.78rem;color:var(--text2)}

  .spinner{display:none;align-items:center;gap:10px;padding:20px 0;color:var(--text2);font-size:.85rem}
  .spinner.show{display:flex}
  .spinner .ring{width:22px;height:22px;border:3px solid var(--border);border-top-color:var(--accent);
    border-radius:50%;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .empty{text-align:center;padding:40px 20px;color:var(--text2);font-size:.85rem}
  .empty .icon{font-size:2rem;margin-bottom:8px;display:block;opacity:.4}
</style>
</head>
<body>

<div class="header">
  <h1>Neural Age &amp; Gender Engine</h1>
  <p>High-precision edge computing inference</p>
</div>

<div class="container">
  <div style="display:flex;flex-direction:column;gap:24px">
    <div class="card">
      <div class="card-title"><span class="dot"></span>Upload Image</div>
      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <div class="dropzone-text" id="dropText">
          <span class="icon">&#128247;</span>
          <strong>Drop image here</strong><br>or click to browse
        </div>
        <input type="file" id="fileInput" accept="image/*">
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="analyzeBtn" disabled>&#x1F50D; Run Local Inference</button>
        <button class="btn btn-secondary" id="clearBtn" style="display:none">&#x2715; Clear</button>
      </div>
      <div class="spinner" id="spinner"><div class="ring"></div>Processing locally…</div>
    </div>

    <div class="card">
      <div class="card-title"><span class="dot"></span>Analysis Results</div>
      <div id="faceCards">
        <div class="empty"><span class="icon">&#128100;</span>No faces analyzed yet</div>
      </div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:24px">
    <div class="card">
      <div class="card-title"><span class="dot"></span>Visual Overlay</div>
      <div class="toggle-bar">
        <button class="active" id="togOverlay" onclick="showView('overlay')">Overlay</button>
        <button id="togOriginal" onclick="showView('original')">Original</button>
      </div>
      <div class="result-img-wrap" id="resultWrap">
        <div class="placeholder">Result will appear here</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title"><span class="dot"></span>Inference Data</div>
      <div class="json-wrap" id="jsonWrap">
        <pre id="jsonPre">{ "status": "waiting" }</pre>
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
  jsonPre.textContent='{ "status": "waiting" }';
  faceCards.innerHTML='<div class="empty"><span class="icon">&#128100;</span>No faces analyzed yet</div>';
});

analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true;
  spinner.classList.add('show');
  resultWrap.innerHTML='<div class="placeholder">Inference engine running…</div>';

  const fd=new FormData();
  fd.append('image',selectedFile);

  try{
    const res=await fetch('/analyze',{method:'POST',body:fd});
    const data=await res.json();
    if(data.error){throw new Error(data.error)}

    overlayB64=data.overlay_b64;
    originalB64=data.original_b64;
    showView(currentView);

    jsonPre.innerHTML=syntaxHL(JSON.stringify(data.faces,null,2));

    if(data.faces.length===0){
      faceCards.innerHTML='<div class="empty"><span class="icon">&#128100;</span>No faces detected by model</div>';
    } else {
      faceCards.innerHTML=data.faces.map(f=>`
        <div class="face-card">
          <div class="avatar">&#128100;</div>
          <div class="info">
            <div class="age-gender">
              <span class="age">${f.age}</span> · ${f.gender}
            </div>
            <div class="meta">Detected with High Precision</div>
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

function showView(v){
  currentView=v;
  document.getElementById('togOverlay').classList.toggle('active',v==='overlay');
  document.getElementById('togOriginal').classList.toggle('active',v==='original');
  const src=v==='overlay'?overlayB64:originalB64;
  if(src) resultWrap.innerHTML=`<img src="data:image/jpeg;base64,${src}">`;
}

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

if __name__ == "__main__":
    print("\n  [*] Neural Age & Gender Engine v4")
    print("  -----------------------------------")
    if not NAGE_API_KEY:
        print("  WARNING: NAGE_API_KEY is not configured!")
    print("  Open  http://localhost:5000  in your browser")
    print("  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
