import io
from PIL import Image
from transformers import pipeline

_model = None

def get_model():
    global _model
    if _model is None:
        _model = pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector"
        )
    return _model

def predict_image(image_bytes: bytes) -> dict:
    """
    바이트 형태의 이미지를 입력받아 AI 생성 여부를 예측합니다.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    model = get_model()
    results = model(image)

    # results 예시: [{'label': 'artificial', 'score': 0.97}, {'label': 'real', 'score': 0.03}]
    # 가장 높은 점수의 레이블로 판별
    top = max(results, key=lambda x: x["score"])
    label_upper = top["label"].upper()
    is_ai = any(k in label_upper for k in ("AI", "FAKE", "ARTIFICIAL", "GENERATED", "SYNTHETIC"))
    confidence = round(top["score"], 4)

    return {
        "status": "success",
        "is_ai_generated": is_ai,
        "confidence": confidence,
        "metadata": {
            "width": width,
            "height": height
        }
    }
