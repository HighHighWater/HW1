from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from app.model import predict_image

app = FastAPI(
    title="AI Image Detector API",
    description="AI 생성 여부 판별 + 이미지 카테고리 분류 API",
    version="2.0.0"
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
      display: flex; align-items: center; justify-content: center;
      padding: 24px; color: #f0f0f0;
    }
    .card {
      background: rgba(255,255,255,0.07);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 24px; padding: 40px;
      width: 100%; max-width: 580px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    }
    .header { text-align: center; margin-bottom: 32px; }
    .header .icon { font-size: 2.8rem; margin-bottom: 10px; }
    .header h1 { font-size: 1.7rem; font-weight: 700; }
    .header p { color: rgba(255,255,255,0.5); font-size: 0.88rem; margin-top: 6px; }

    .drop-zone {
      border: 2px dashed rgba(255,255,255,0.2); border-radius: 16px;
      padding: 36px 20px; text-align: center; cursor: pointer;
      transition: all 0.25s ease; background: rgba(255,255,255,0.03); position: relative;
    }
    .drop-zone:hover, .drop-zone.drag-over {
      border-color: #a78bfa; background: rgba(167,139,250,0.08);
    }
    .drop-zone input { display: none; }
    .drop-zone .dz-icon { font-size: 2.4rem; color: #a78bfa; margin-bottom: 12px; }
    .drop-zone .dz-text { font-size: 0.95rem; color: rgba(255,255,255,0.6); }
    .drop-zone .dz-sub  { font-size: 0.78rem; color: rgba(255,255,255,0.35); margin-top: 6px; }
    .drop-zone .dz-btn  {
      display: inline-block; margin-top: 14px; padding: 8px 20px;
      background: rgba(167,139,250,0.2); border: 1px solid rgba(167,139,250,0.4);
      border-radius: 999px; font-size: 0.82rem; color: #c4b5fd;
    }

    #preview-wrap { display: none; margin-top: 20px; border-radius: 14px; overflow: hidden; position: relative; }
    #preview-wrap img { width: 100%; max-height: 240px; object-fit: cover; display: block; }
    #preview-wrap .change-btn {
      position: absolute; top: 10px; right: 10px;
      background: rgba(0,0,0,0.6); border: none; color: white;
      border-radius: 999px; padding: 5px 12px; font-size: 0.78rem; cursor: pointer;
    }

    #analyze-btn {
      display: none; width: 100%; margin-top: 20px; padding: 14px; border: none;
      border-radius: 14px; background: linear-gradient(90deg, #7c3aed, #4f46e5);
      color: white; font-size: 1rem; font-weight: 600; cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }
    #analyze-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    #analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    #result { display: none; margin-top: 24px; }

    /* AI 판별 결과 */
    .result-verdict {
      border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 16px;
    }
    .result-verdict.is-ai {
      background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(79,70,229,0.15));
      border: 1px solid rgba(139,92,246,0.4);
    }
    .result-verdict.is-real {
      background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
      border: 1px solid rgba(16,185,129,0.35);
    }
    .verdict-icon { font-size: 2.2rem; margin-bottom: 6px; }
    .verdict-label { font-size: 1.2rem; font-weight: 700; }
    .verdict-sub { font-size: 0.8rem; color: rgba(255,255,255,0.5); margin-top: 4px; }

    /* 원형 게이지 */
    .gauge-wrap { margin: 0 auto 4px; width: 130px; height: 130px; position: relative; }
    svg.gauge { transform: rotate(-90deg); }
    .gauge-bg   { fill: none; stroke: rgba(255,255,255,0.08); stroke-width: 10; }
    .gauge-fill { fill: none; stroke-width: 10; stroke-linecap: round; transition: stroke-dashoffset 1s ease; }
    .gauge-center {
      position: absolute; inset: 0;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .gauge-pct { font-size: 1.7rem; font-weight: 800; line-height: 1; }
    .gauge-lbl { font-size: 0.68rem; color: rgba(255,255,255,0.45); margin-top: 2px; }

    /* 카테고리 카드 */
    .category-card {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 18px 20px;
      display: flex; align-items: center; gap: 16px;
      margin-top: 16px;
    }
    .cat-icon { font-size: 2.6rem; line-height: 1; }
    .cat-info { flex: 1; }
    .cat-label { font-size: 0.72rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; }
    .cat-name  { font-size: 1.3rem; font-weight: 700; margin: 2px 0 6px; }
    .cat-raw   { font-size: 0.75rem; color: rgba(255,255,255,0.35); }
    .cat-bar-wrap { margin-top: 6px; }
    .cat-bar-bg {
      height: 6px; background: rgba(255,255,255,0.08);
      border-radius: 999px; overflow: hidden;
    }
    .cat-bar-fill {
      height: 100%; border-radius: 999px;
      background: linear-gradient(90deg, #f59e0b, #f97316);
      transition: width 1s ease; width: 0%;
    }
    .cat-conf { font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 3px; text-align: right; }

    /* 메타 */
    .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
    .meta-item { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 10px 14px; }
    .mi-label  { font-size: 0.7rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; }
    .mi-value  { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }

    .spinner {
      display: inline-block; width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.3); border-top-color: white;
      border-radius: 50%; animation: spin 0.7s linear infinite;
      vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    #error-msg {
      display: none; margin-top: 16px; padding: 12px 16px;
      background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.35);
      border-radius: 10px; font-size: 0.85rem; color: #fca5a5;
    }
  </style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="icon">🤖</div>
    <h1>AI Image Detector</h1>
    <p>AI 생성 여부 판별 · 이미지 카테고리 분류</p>
  </div>

  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="image/*"/>
    <div class="dz-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
    <div class="dz-text">이미지를 여기로 드래그하거나</div>
    <div class="dz-sub">PNG · JPG · WEBP · GIF 지원</div>
    <div class="dz-btn">파일 선택</div>
  </div>

  <div id="preview-wrap">
    <img id="preview-img" src="" alt="preview"/>
    <button class="change-btn" onclick="resetAll()"><i class="fa-solid fa-xmark"></i> 변경</button>
  </div>

  <button id="analyze-btn" onclick="analyze()">
    <i class="fa-solid fa-magnifying-glass"></i>&nbsp; 분석 시작
  </button>

  <div id="error-msg"></div>

  <div id="result">
    <!-- AI 판별 -->
    <div class="result-verdict" id="verdict-box">
      <div class="verdict-icon" id="verdict-icon"></div>
      <div class="verdict-label" id="verdict-label"></div>
      <div class="verdict-sub" id="verdict-sub"></div>
    </div>

    <!-- 확신도 게이지 -->
    <div style="text-align:center; margin-bottom:4px;">
      <div class="gauge-wrap">
        <svg class="gauge" viewBox="0 0 100 100" width="130" height="130">
          <circle class="gauge-bg" cx="50" cy="50" r="40"/>
          <circle class="gauge-fill" id="gauge-fill" cx="50" cy="50" r="40"
            stroke-dasharray="251.2" stroke-dashoffset="251.2"/>
        </svg>
        <div class="gauge-center">
          <span class="gauge-pct" id="gauge-pct">0%</span>
          <span class="gauge-lbl">AI 확신도</span>
        </div>
      </div>
    </div>

    <!-- 카테고리 분류 -->
    <div class="category-card" id="category-card">
      <div class="cat-icon" id="cat-icon"></div>
      <div class="cat-info">
        <div class="cat-label">이미지 카테고리</div>
        <div class="cat-name" id="cat-name"></div>
        <div class="cat-raw" id="cat-raw"></div>
        <div class="cat-bar-wrap">
          <div class="cat-bar-bg">
            <div class="cat-bar-fill" id="cat-bar"></div>
          </div>
          <div class="cat-conf" id="cat-conf"></div>
        </div>
      </div>
    </div>

    <!-- 메타 -->
    <div class="meta-grid">
      <div class="meta-item">
        <div class="mi-label">파일명</div>
        <div class="mi-value" id="meta-filename" style="font-size:0.78rem;word-break:break-all;">-</div>
      </div>
      <div class="meta-item">
        <div class="mi-label">해상도</div>
        <div class="mi-value" id="meta-size">-</div>
      </div>
    </div>
  </div>
</div>

<script>
  const dropZone  = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  let selectedFile = null;

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) loadFile(f);
  });

  function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      document.getElementById('preview-img').src = ev.target.result;
      dropZone.style.display = 'none';
      document.getElementById('preview-wrap').style.display = 'block';
      document.getElementById('analyze-btn').style.display = 'block';
      document.getElementById('result').style.display = 'none';
      document.getElementById('error-msg').style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  function resetAll() {
    selectedFile = null; fileInput.value = '';
    dropZone.style.display = 'block';
    document.getElementById('preview-wrap').style.display = 'none';
    document.getElementById('analyze-btn').style.display = 'none';
    document.getElementById('result').style.display = 'none';
    document.getElementById('error-msg').style.display = 'none';
  }

  async function analyze() {
    if (!selectedFile) return;
    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>분석 중... (첫 실행 시 모델 로딩으로 수십 초 소요)';
    document.getElementById('error-msg').style.display = 'none';
    document.getElementById('result').style.display = 'none';

    const form = new FormData();
    form.append('file', selectedFile);
    try {
      const res  = await fetch('/predict/', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '오류가 발생했습니다.');
      showResult(data);
    } catch(err) {
      document.getElementById('error-msg').textContent = '오류: ' + err.message;
      document.getElementById('error-msg').style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i>&nbsp; 다시 분석';
    }
  }

  function showResult(data) {
    const p = data.prediction;
    const isAI = p.is_ai_generated;
    const conf = Math.round(p.confidence * 100);
    const cat  = p.category;

    // 판별 결과
    document.getElementById('verdict-box').className = 'result-verdict ' + (isAI ? 'is-ai' : 'is-real');
    document.getElementById('verdict-icon').textContent  = isAI ? '🤖' : '📷';
    document.getElementById('verdict-label').textContent = isAI ? 'AI 생성 이미지' : '실제 사진';
    document.getElementById('verdict-sub').textContent   = isAI
      ? 'AI가 생성한 이미지로 판단됩니다'
      : '실제 카메라로 촬영된 사진으로 판단됩니다';

    // 확신도 게이지
    const color = isAI ? '#a78bfa' : '#10b981';
    const fill  = document.getElementById('gauge-fill');
    fill.style.stroke = color;
    fill.style.strokeDashoffset = 251.2 - (conf / 100) * 251.2;
    let cur = 0;
    const pctEl = document.getElementById('gauge-pct');
    pctEl.style.color = color;
    const iv = setInterval(() => {
      cur = Math.min(cur + Math.ceil(conf / 25), conf);
      pctEl.textContent = cur + '%';
      if (cur >= conf) clearInterval(iv);
    }, 30);

    // 카테고리
    document.getElementById('cat-icon').textContent = cat.icon;
    document.getElementById('cat-name').textContent = cat.name;
    document.getElementById('cat-raw').textContent  = cat.raw_label;
    const catConf = Math.round(cat.confidence * 100);
    document.getElementById('cat-conf').textContent = catConf + '%';
    setTimeout(() => {
      document.getElementById('cat-bar').style.width = catConf + '%';
    }, 100);

    // 메타
    document.getElementById('meta-filename').textContent = data.filename;
    document.getElementById('meta-size').textContent =
      p.metadata ? p.metadata.width + ' × ' + p.metadata.height : '-';

    document.getElementById('result').style.display = 'block';
    document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
