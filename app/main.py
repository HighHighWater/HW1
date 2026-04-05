from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.model import predict_image

app = FastAPI(
    title="AI Image Detector API",
    description="이미지가 AI로 생성된 것인지 실제 사람/카메라가 찍은 것인지 판별하는 간단한 API 서버입니다.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "AI Image Detector API 서버가 정상적으로 실행 중입니다. /docs 에서 API 문서를 확인하세요."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 파일 확장자 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 파일 읽기
        contents = await file.read()
        
        # 모델 예측
        result = predict_image(contents)
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
