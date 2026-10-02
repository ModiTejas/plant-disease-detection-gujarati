from io import BytesIO
import json
import traceback
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

try:
    from API.vision_service import analyze_plant_with_vision, is_plant_image
except ImportError:
    from vision_service import analyze_plant_with_vision, is_plant_image

# ---------------- PATHS & CONSTANTS ----------------
MODEL_PTH = BASE_DIR / "models" / "best_model.pth"
MODEL_ONNX = BASE_DIR / "models" / "best_model.onnx"
CLASS_INDICES_PATH = BASE_DIR / "models" / "class_indices.json"
DICT_PATH = BASE_DIR / "disease_dictionary.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

EXPECTED_CLASS_COUNT = 37
CONFIDENCE_THRESHOLD = 0.60
IMAGE_SIZE = (224, 224)

app = FastAPI(title="Plant Disease Dual-Mode API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Classes & Dictionary
with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)
    CLASSES = class_data["classes"]

with open(DICT_PATH, "r", encoding="utf-8") as f:
    try:
        raw_dict = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse disease dictionary JSON: {DICT_PATH}") from exc

if not isinstance(raw_dict, dict) or "classes" not in raw_dict:
    raise ValueError("Disease dictionary must contain a 'classes' list or dictionary.")

classes_data = raw_dict["classes"]
DISEASE_DICT = {}
seen_class_ids = set()

if isinstance(classes_data, list):
    for item in classes_data:
        if not isinstance(item, dict) or "class_id" not in item:
            raise ValueError("Disease dictionary contains a class without class_id.")
        class_id = item["class_id"]
        if not isinstance(class_id, str) or not class_id.strip():
            raise ValueError("Disease dictionary contains an empty or invalid class_id.")
        lookup_key = class_id.strip().lower()
        if lookup_key in seen_class_ids:
            raise ValueError(f"Disease dictionary contains duplicate class_id: {class_id}")
        seen_class_ids.add(lookup_key)
        DISEASE_DICT[lookup_key] = item
elif isinstance(classes_data, dict):
    for class_id, item in classes_data.items():
        if not isinstance(class_id, str) or not class_id.strip():
            raise ValueError("Disease dictionary contains an empty or invalid class_id.")
        if not isinstance(item, dict):
            raise ValueError(f"Disease dictionary entry for {class_id} must be an object.")
        lookup_key = class_id.strip().lower()
        if lookup_key in seen_class_ids:
            raise ValueError(f"Disease dictionary contains duplicate class_id: {class_id}")
        seen_class_ids.add(lookup_key)
        DISEASE_DICT[lookup_key] = item
else:
    raise ValueError("Disease dictionary 'classes' must be a list or dictionary.")

if len(DISEASE_DICT) != EXPECTED_CLASS_COUNT:
    raise ValueError(
        f"Disease dictionary must contain exactly {EXPECTED_CLASS_COUNT} classes; "
        f"found {len(DISEASE_DICT)}."
    )

def normalize_class_name(class_name: str) -> str:
    return " ".join(class_name.replace("_", " ").replace("-", " ").split()).lower()

LEGACY_CLASS_LOOKUP = {
    normalize_class_name(item.get("class_id", key)): key
    for key, item in DISEASE_DICT.items()
}
CLASS_NAME_ALIASES = {
    normalize_class_name("Cotton___Healthy Leaf"): normalize_class_name("Cotton___Healthy"),
    normalize_class_name("Cotton___Leaf Redding"): normalize_class_name("Cotton___Leaf Reddening"),
}

def _as_text_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []

def get_disease_details(class_name: str) -> dict:
    class_id = class_name.strip()
    details = DISEASE_DICT.get(class_id.lower())
    if details is None:
        legacy_key = normalize_class_name(class_id)
        legacy_key = CLASS_NAME_ALIASES.get(legacy_key, legacy_key)
        lookup_key = LEGACY_CLASS_LOOKUP.get(legacy_key)
        details = DISEASE_DICT.get(lookup_key, {})
    sections = details.get("farmer_result", {}).get("sections", {})
    if not sections:
        return details

    chemical = sections.get("chemical_control", {})
    biological = sections.get("natural_biological_control", {})
    current_steps = sections.get("what_to_do_now", {})
    prevention = sections.get("prevention", {})
    management = (
        _as_text_list(chemical.get("recommended_active_ingredients"))
        + _as_text_list(chemical.get("description"))
        + _as_text_list(biological.get("recommendations"))
        + _as_text_list(current_steps.get("steps"))
    )

    return {
        **details,
        "plant_gu": details.get("crop_name_gu", ""),
        "name_gu": details.get("condition_name_gu", ""),
        "type_gu": details.get("category_gu", ""),
        "cause_gu": details.get("pathogen_scientific_name", ""),
        "symptoms_gu": sections.get("what_happened", ""),
        "management_gu": management,
        "prevention_gu": _as_text_list(prevention.get("steps")),
    }

# 2. Setup Inference Engine (ONNX for Vercel/Production, fallback to PyTorch)
USE_ONNX = MODEL_ONNX.exists()
ort_session = None

if USE_ONNX:
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(MODEL_ONNX), providers=['CPUExecutionProvider'])
    print("Inference Engine: ONNX Runtime (Vercel-Optimized)")
else:
    import torch
    import torch.nn as nn
    from torchvision import models
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    ckpt = torch.load(MODEL_PTH, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Inference Engine: PyTorch CPU")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    img_arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_arr = np.transpose(img_arr, (2, 0, 1))  # (C, H, W)
    return np.expand_dims(img_arr, axis=0)      # (1, C, H, W)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ---------------- ENDPOINTS ----------------
@app.get("/ping")
async def ping():
    return {"status": "live", "engine": "ONNX" if USE_ONNX else "PyTorch", "total_classes": len(CLASSES)}

@app.post("/predict")
async def predict_project_model(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="કૃપા કરીને માન્ય ફોટો ફાઇલ અપલોડ કરો.")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="ફોટો ફાઇલ વાંચવામાં અસમર્થ.")

    try:
        if not await run_in_threadpool(is_plant_image, image):
            return {
                "mode": "project",
                "status": "success",
                "is_plant": False,
                "error_gu": "⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ દેખાતું નથી. કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."
            }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="છોડની તસવીર ચકાસી શકાઈ નથી. કૃપા કરીને GEMINI_API_KEY તપાસી ફરી પ્રયાસ કરો."
        ) from e

    def _run_inference() -> np.ndarray:
        input_data = preprocess_image(image)
        if USE_ONNX:
            outputs = ort_session.run(None, {'input': input_data})[0]
            return softmax(outputs)[0]
        with torch.no_grad():
            outputs = model(torch.from_numpy(input_data)).numpy()
            return softmax(outputs)[0]

    probabilities = await run_in_threadpool(_run_inference)

    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])
    predicted_class = CLASSES[top_idx]

    is_confident = confidence >= CONFIDENCE_THRESHOLD
    details = get_disease_details(predicted_class)

    top3_indices = np.argsort(probabilities)[::-1][:3]
    top_3 = [
        {
            "class": CLASSES[idx],
            "predicted_class": CLASSES[idx],
            "name_gu": get_disease_details(CLASSES[idx]).get("name_gu") or CLASSES[idx],
            "confidence": round(float(probabilities[idx]) * 100, 2)
        }
        for idx in top3_indices
    ]

    return {
        "mode": "project",
        "status": "success",
        "is_confident": is_confident,
        "is_uncertain": not is_confident,
        "prediction": predicted_class,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "details": details,
        "top_3": top_3,
        "top_predictions": top_3
    }

@app.post("/analyze-vision")
async def predict_vision_ai(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="કૃપા કરીને માન્ય ફોટો ફાઇલ અપલોડ કરો.")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="ફોટો ફાઇલ વાંચવામાં અસમર્થ.")

    try:
        diagnosis = await run_in_threadpool(analyze_plant_with_vision, image)
        return {"mode": "vision", "status": "success", "diagnosis": diagnosis}
    except Exception as e:
        error_msg = str(e)
        status_code = 429 if "429" in error_msg or "rate limit" in error_msg.lower() else 500
        raise HTTPException(status_code=status_code, detail=f"AI વિઝન: {error_msg}")

@app.get("/")
async def serve_index():
    return FileResponse(INDEX_FILE)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    uvicorn.run("API.main:app", host="127.0.0.1", port=8000, reload=True)
