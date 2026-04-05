import io
from PIL import Image
# from transformers import pipeline

# Senior DevOps Engineer Tip:
# MLOps 파이프라인 초기 단계에서는 실제 무거운 모델을 바로 로드하기보다는,
# 인터페이스를 먼저 맞추고 나중에 파이프라인(CI/CD, 모델 레지스트리 등)이 갖춰진 뒤 
# 실제 모델을 주입(inject)하는 방식을 추천합니다.
# 
# 실제 가벼운 모델을 사용하려면 Hugging Face의 transformers를 사용할 수 있습니다.
# 예시:
# def load_real_model():
#     # 로컬 또는 Hugging Face Model Hub에서 가벼운 AI 탐지 모델(MobileNet, ViT 기반 등)을 로드합니다.
#     return pipeline("image-classification", model="Nahid/ai-generated-image-detector")
# 
# model = load_real_model()

def predict_image(image_bytes: bytes) -> dict:
    """
    바이트 형태의 이미지를 입력받아 AI 생성 여부를 예측합니다.
    """
    # 이미지 열기 검증 (형식 오류 등을 잡기 위함)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # ---------------------------------------------------------
    # 여기에 실제 모델 추론 로직이 들어갑니다.
    # 예: 
    # result = model(image)
    # return {"is_ai_generated": result[0]['label'] == 'AI', "confidence": result[0]['score']}
    # ---------------------------------------------------------

    # 현재는 MLOps 서버/인프라 구성 테스트를 위한 모의(Mock) 응답을 반환합니다.
    # 이미지 사이즈를 기반으로 간단한 로그를 남기거나 처리할 수도 있습니다.
    width, height = image.size
    
    return {
        "status": "success",
        "is_ai_generated": False,  # True if AI generated, False if Real
        "confidence": 0.88,        # 모델의 확신도
        "metadata": {
            "width": width,
            "height": height
        },
        "note": "이 응답은 현재 인프라 테스트용 Mock 데이터입니다. app/model.py에서 실제 모델로 교체하세요."
    }
