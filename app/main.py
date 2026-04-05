from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from app.model import predict_image

app = FastAPI(
    title="AI Image Detector API",
    description="이미지가 AI로 생성된 것인지 실제 사람/카메라가 찍은 것인지 판별하는 API 서버입니다.",
    version="1.0.0"
)

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Image Detector</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      min-height: 100vh;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      color: #f0f0f0;
    }

    .card {
      background: rgba(255,255,255,0.07);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 24px;
      padding: 40px;
      width: 100%;
      max-width: 560px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    }

    .header { text-align: center; margin-bottom: 32px; }
    .header .icon { font-size: 2.8rem; margin-bottom: 10px; }
    .header h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; }
    .header p { color: rgba(255,255,255,0.5); font-size: 0.88rem; margin-top: 6px; }

    /* Drop Zone */
    .drop-zone {
      border: 2px dashed rgba(255,255,255,0.2);
      border-radius: 16px;
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
      background: rgba(255,255,255,0.03);
    }
    .drop-zone:hover, .drop-zone.drag-over {
      border-color: #a78bfa;
      background: rgba(167,139,250,0.08);
    }
    .drop-zone input { display: none; }
    .drop-zone .dz-icon { font-size: 2.4rem; color: #a78bfa; margin-bottom: 12px; }
    .drop-zone .dz-text { font-size: 0.95rem; color: rgba(255,255,255,0.6); }
    .drop-zone .dz-sub  { font-size: 0.78rem; color: rgba(255,255,255,0.35); margin-top: 6px; }
    .drop-zone .dz-btn  {
      display: inline-block;
      margin-top: 14px;
      padding: 8px 20px;
      background: rgba(167,139,250,0.2);
      border: 1px solid rgba(167,139,250,0.4);
      border-radius: 999px;
      font-size: 0.82rem;
      color: #c4b5fd;
      transition: background 0.2s;
    }
    .drop-zone:hover .dz-btn { background: rgba(167,139,250,0.35); }

    /* Preview */
    #preview-wrap {
      display: none;
      margin-top: 20px;
      border-radius: 14px;
      overflow: hidden;
      position: relative;
    }
    #preview-wrap img {
      width: 100%;
      max-height: 260px;
      object-fit: cover;
      display: block;
      border-radius: 14px;
    }
    #preview-wrap .change-btn {
      position: absolute;
      top: 10px; right: 10px;
      background: rgba(0,0,0,0.6);
      border: none;
      color: white;
      border-radius: 999px;
      padding: 5px 12px;
      font-size: 0.78rem;
      cursor: pointer;
    }

    /* Analyze button */
    #analyze-btn {
      display: none;
      width: 100%;
      margin-top: 20px;
      padding: 14px;
      border: none;
      border-radius: 14px;
      background: linear-gradient(90deg, #7c3aed, #4f46e5);
      color: white;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
      letter-spacing: 0.3px;
    }
    #analyze-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    #analyze-btn:active { transform: translateY(0); }
    #analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Result card */
    #result { display: none; margin-top: 24px; }

    .result-verdict {
      border-radius: 16px;
      padding: 24px;
      text-align: center;
      margin-bottom: 20px;
    }
    .result-verdict.is-ai {
      background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(79,70,229,0.15));
      border: 1px solid rgba(139,92,246,0.4);
    }
    .result-verdict.is-real {
      background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
      border: 1px solid rgba(16,185,129,0.35);
    }
    .verdict-icon { font-size: 2.6rem; margin-bottom: 8px; }
    .verdict-label { font-size: 1.25rem; font-weight: 700; }
    .verdict-sub { font-size: 0.82rem; color: rgba(255,255,255,0.5); margin-top: 4px; }

    /* Confidence gauge */
    .gauge-wrap { margin: 0 auto 4px; width: 140px; height: 140px; position: relative; }
    svg.gauge { transform: rotate(-90deg); }
    .gauge-bg { fill: none; stroke: rgba(255,255,255,0.08); stroke-width: 10; }
    .gauge-fill { fill: none; stroke-width: 10; stroke-linecap: round; transition: stroke-dashoffset 1s ease; }
    .gauge-center {
      position: absolute; inset: 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
    }
    .gauge-pct { font-size: 1.8rem; font-weight: 800; line-height: 1; }
    .gauge-lbl { font-size: 0.7rem; color: rgba(255,255,255,0.45); margin-top: 2px; }

    /* Metadata */
    .meta-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 16px;
    }
    .meta-item {
      background: rgba(255,255,255,0.05);
      border-radius: 10px;
      padding: 12px 14px;
    }
    .meta-item .mi-label { font-size: 0.72rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; }
    .meta-item .mi-value { font-size: 0.95rem; font-weight: 600; margin-top: 3px; }

    /* Loading spinner */
    .spinner {
      display: inline-block;
      width: 18px; height: 18px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Error */
    #error-msg {
      display: none;
      margin-top: 16px;
      padding: 12px 16px;
      background: rgba(239,68,68,0.15);
      border: 1px solid rgba(239,68,68,0.35);
      border-radius: 10px;
      font-size: 0.85rem;
      color: #fca5a5;
    }
  </style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="icon">🤖</div>
    <h1>AI Image Detector</h1>
    <p>이미지가 AI로 생성됐는지, 실제 사진인지 판별해드립니다</p>
  </div>

  <!-- Drop Zone -->
  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="image/*"/>
    <div class="dz-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
    <div class="dz-text">이미지를 여기로 드래그하거나</div>
    <div class="dz-sub">PNG · JPG · WEBP · GIF 지원</div>
    <div class="dz-btn">파일 선택</div>
  </div>

  <!-- Preview -->
  <div id="preview-wrap">
    <img id="preview-img" src="" alt="preview"/>
    <button class="change-btn" onclick="resetAll()"><i class="fa-solid fa-xmark"></i> 변경</button>
  </div>

  <!-- Analyze -->
  <button id="analyze-btn" onclick="analyze()">
    <i class="fa-solid fa-magnifying-glass"></i>&nbsp; 분석 시작
  </button>

  <!-- Error -->
  <div id="error-msg"></div>

  <!-- Result -->
  <div id="result">
    <div class="result-verdict" id="verdict-box">
      <div class="verdict-icon" id="verdict-icon"></div>
      <div class="verdict-label" id="verdict-label"></div>
      <div class="verdict-sub" id="verdict-sub"></div>
    </div>

    <div style="text-align:center;">
      <div class="gauge-wrap">
        <svg class="gauge" viewBox="0 0 100 100" width="140" height="140">
          <circle class="gauge-bg" cx="50" cy="50" r="40"/>
          <circle class="gauge-fill" id="gauge-fill" cx="50" cy="50" r="40"
            stroke-dasharray="251.2" stroke-dashoffset="251.2"/>
        </svg>
        <div class="gauge-center">
          <span class="gauge-pct" id="gauge-pct">0%</span>
          <span class="gauge-lbl">확신도</span>
        </div>
      </div>
    </div>

    <div class="meta-grid">
      <div class="meta-item">
        <div class="mi-label">파일명</div>
        <div class="mi-value" id="meta-filename" style="font-size:0.8rem;word-break:break-all;">-</div>
      </div>
      <div class="meta-item">
        <div class="mi-label">해상도</div>
        <div class="mi-value" id="meta-size">-</div>
      </div>
    </div>
  </div>
</div>

<script>
  const dropZone   = document.getElementById('drop-zone');
  const fileInput  = document.getElementById('file-input');
  const previewWrap= document.getElementById('preview-wrap');
  const previewImg = document.getElementById('preview-img');
  const analyzeBtn = document.getElementById('analyze-btn');
  const resultDiv  = document.getElementById('result');
  const errorDiv   = document.getElementById('error-msg');

  let selectedFile = null;

  // Click to open file dialog
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });

  // Drag & Drop
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) loadFile(f);
  });

  function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.src = ev.target.result;
      dropZone.style.display  = 'none';
      previewWrap.style.display = 'block';
      analyzeBtn.style.display  = 'block';
      resultDiv.style.display   = 'none';
      errorDiv.style.display    = 'none';
    };
    reader.readAsDataURL(file);
  }

  function resetAll() {
    selectedFile = null;
    fileInput.value = '';
    dropZone.style.display    = 'block';
    previewWrap.style.display = 'none';
    analyzeBtn.style.display  = 'none';
    resultDiv.style.display   = 'none';
    errorDiv.style.display    = 'none';
  }

  async function analyze() {
    if (!selectedFile) return;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner"></span>분석 중...';
    errorDiv.style.display = 'none';
    resultDiv.style.display = 'none';

    const form = new FormData();
    form.append('file', selectedFile);

    try {
      const res  = await fetch('/predict/', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '오류가 발생했습니다.');
      showResult(data);
    } catch(err) {
      errorDiv.textContent = '오류: ' + err.message;
      errorDiv.style.display = 'block';
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i>&nbsp; 다시 분석';
    }
  }

  function showResult(data) {
    const p = data.prediction;
    const isAI = p.is_ai_generated;
    const conf = Math.round(p.confidence * 100);

    // Verdict box
    const box = document.getElementById('verdict-box');
    box.className = 'result-verdict ' + (isAI ? 'is-ai' : 'is-real');
    document.getElementById('verdict-icon').textContent  = isAI ? '🤖' : '📷';
    document.getElementById('verdict-label').textContent = isAI ? 'AI 생성 이미지' : '실제 사진';
    document.getElementById('verdict-sub').textContent   = isAI
      ? 'AI가 만들어낸 이미지로 판단됩니다'
      : '실제 카메라로 촬영된 사진으로 판단됩니다';

    // Gauge
    const color = isAI ? '#a78bfa' : '#10b981';
    const fill  = document.getElementById('gauge-fill');
    fill.style.stroke = color;
    const circumference = 251.2;
    const offset = circumference - (conf / 100) * circumference;
    requestAnimationFrame(() => {
      fill.style.strokeDashoffset = offset;
    });

    // Count up
    let current = 0;
    const pctEl = document.getElementById('gauge-pct');
    pctEl.style.color = color;
    const interval = setInterval(() => {
      current = Math.min(current + Math.ceil(conf / 30), conf);
      pctEl.textContent = current + '%';
      if (current >= conf) clearInterval(interval);
    }, 30);

    // Meta
    document.getElementById('meta-filename').textContent = data.filename;
    document.getElementById('meta-size').textContent =
      p.metadata ? `${p.metadata.width} × ${p.metadata.height}` : '-';

    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    try:
        contents = await file.read()
        result = predict_image(contents)
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
