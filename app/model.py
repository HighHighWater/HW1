import io
from PIL import Image
from transformers import pipeline

_ai_detector = None
_classifier = None

# ImageNet 1000개 레이블 → 4개 카테고리 매핑 키워드
PERSON_KEYWORDS = {
    "person", "man", "woman", "boy", "girl", "people", "human",
    "player", "nurse", "doctor", "soldier", "baseball", "basketball",
    "scuba", "bridegroom", "bride"
}
ANIMAL_KEYWORDS = {
    "dog", "cat", "bird", "fish", "horse", "cow", "elephant", "tiger",
    "lion", "bear", "wolf", "fox", "rabbit", "snake", "frog", "turtle",
    "shark", "whale", "insect", "bee", "butterfly", "spider", "ant",
    "cheetah", "zebra", "giraffe", "panda", "monkey", "gorilla", "parrot",
    "penguin", "flamingo", "eagle", "hawk", "owl", "hamster", "squirrel",
    "deer", "camel", "seal", "otter", "crab", "lobster", "jellyfish",
    "hen", "rooster", "turkey", "goose", "duck", "pig", "sheep", "goat"
}
LANDSCAPE_KEYWORDS = {
    "mountain", "beach", "ocean", "sea", "lake", "river", "forest",
    "field", "sky", "desert", "valley", "cliff", "waterfall", "volcano",
    "coral", "alp", "promontory", "sandbar", "seashore", "lakeside",
    "geyser", "stone wall", "cliff dwelling"
}


def get_ai_detector():
    global _ai_detector
    if _ai_detector is None:
        _ai_detector = pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector"
        )
    return _ai_detector


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
        )
    return _classifier


def map_to_category(label: str) -> tuple:
    label_lower = label.lower()
    if any(k in label_lower for k in PERSON_KEYWORDS):
        return "사람", "🧍"
    if any(k in label_lower for k in ANIMAL_KEYWORDS):
        return "동물", "🐾"
    if any(k in label_lower for k in LANDSCAPE_KEYWORDS):
        return "풍경", "🏔️"
    return "사물", "📦"


def predict_image(image_bytes: bytes) -> dict:
    """
    AI 생성 여부 판별 + 이미지 카테고리 분류 (사람/동물/풍경/사물)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # 1. AI 생성 여부 판별
    ai_results = get_ai_detector()(image)
    top_ai = max(ai_results, key=lambda x: x["score"])
    label_upper = top_ai["label"].upper()
    is_ai = any(k in label_upper for k in ("AI", "FAKE", "ARTIFICIAL", "GENERATED", "SYNTHETIC"))
    ai_confidence = round(top_ai["score"], 4)

    # 2. 이미지 카테고리 분류
    class_results = get_classifier()(image)
    top_class = class_results[0]
    category_name, category_icon = map_to_category(top_class["label"])

    return {
        "status": "success",
        "is_ai_generated": is_ai,
        "confidence": ai_confidence,
        "category": {
            "name": category_name,
            "icon": category_icon,
            "raw_label": top_class["label"],
            "confidence": round(top_class["score"], 4)
        },
        "metadata": {
            "width": width,
            "height": height
        }
    }
