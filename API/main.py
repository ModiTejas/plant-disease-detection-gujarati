from io import BytesIO
import json
import traceback
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
if not (BASE_DIR / ".env").exists():
    BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

try:
    from API.vision_service import analyze_plant_with_vision
except ImportError:
    try:
        from api.vision_service import analyze_plant_with_vision
    except ImportError:
        from vision_service import analyze_plant_with_vision

# ---------------- PATHS & CONSTANTS ----------------
MODEL_ONNX = BASE_DIR / "models" / "best_model.onnx"
MODEL_PTH = BASE_DIR / "models" / "best_model.pth"
CLASS_INDICES_PATH = BASE_DIR / "models" / "class_indices.json"
DICT_PATH = BASE_DIR / "disease_dictionary.json"

STATIC_DIR = BASE_DIR / "API" / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = BASE_DIR / "api" / "static"
INDEX_FILE = STATIC_DIR / "index.html"

EXPECTED_CLASS_COUNT = 37
CONFIDENCE_THRESHOLD = 0.60
IMAGE_SIZE = (224, 224)

app = FastAPI(title="Plant Disease Dual-Mode API", version="3.1.0")

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
    raw_dict = json.load(f)
    classes_dict = raw_dict.get("classes", raw_dict)
    DISEASE_DICT = {k.strip().lower(): v for k, v in classes_dict.items()}

def get_disease_details(class_name: str) -> dict:
    return DISEASE_DICT.get(class_name.strip().lower(), {})

# 2. Setup Inference Engine (ONNX for Vercel/Production, PyTorch local)
USE_ONNX = MODEL_ONNX.exists()
ort_session = None

if USE_ONNX:
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(MODEL_ONNX), providers=['CPUExecutionProvider'])
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

# ---------------- BOTANICAL & LEAF FILTER ----------------
def is_botanical_image(image: Image.Image) -> tuple[bool, float]:
    """
    Analyzes HSV color distribution to verify if the image contains
    organic leaf/plant matter (Greens, Chlorotic Yellows, Necrotic Browns).
    """
    img_hsv = image.convert("HSV").resize((120, 120))
    hsv_arr = np.array(img_hsv, dtype=np.float32)
    
    H = hsv_arr[:, :, 0] * (360.0 / 255.0)  # Hue: 0-360 degrees
    S = hsv_arr[:, :, 1] / 255.0            # Saturation: 0-1
    V = hsv_arr[:, :, 2] / 255.0            # Value: 0-1

    # Botanical foliage masks:
    # 1. Healthy green foliage (Hue 60-170, Saturation > 0.15)
    green_mask = (H >= 55) & (H <= 175) & (S >= 0.12) & (V >= 0.10)
    
    # 2. Diseased / Chlorotic yellow foliage (Hue 30-55, Saturation > 0.18)
    yellow_mask = (H >= 28) & (H < 55) & (S >= 0.18) & (V >= 0.18)
    
    # 3. Necrotic brown / blight lesions (Hue 10-28, moderate saturation and value)
    brown_mask = (H >= 8) & (H < 28) & (S >= 0.15) & (V >= 0.10) & (V <= 0.85)

    botanical_pixels = np.sum(green_mask | yellow_mask | brown_mask)
    total_pixels = 120 * 120
    botanical_ratio = float(botanical_pixels / total_pixels)

    # If at least 10% of pixels match organic leaf/foliage color ranges, it is accepted
    is_plant = botanical_ratio >= 0.10
    return is_plant, round(botanical_ratio * 100, 1)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    img_arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_arr = np.transpose(img_arr, (2, 0, 1))
    return np.expand_dims(img_arr, axis=0)

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

    # Step 1: Pre-filter - Check if image contains plant/leaf matter
    is_plant, foliage_pct = is_botanical_image(image)
    if not is_plant:
        return {
            "mode": "project",
            "status": "success",
            "is_plant": False,
            "foliage_density": foliage_pct,
            "error_gu": "આ તસવીરમાં કોઈ છોડ કે પાન ઓળખાયું નથી. કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો.",
            "prediction": "Not a Plant / Leaf",
            "confidence": 0.0
        }

    # Step 2: Run Inference
    input_data = preprocess_image(image)

    if USE_ONNX:
        outputs = ort_session.run(None, {'input': input_data})[0]
        probabilities = softmax(outputs)[0]
    else:
        with torch.no_grad():
            outputs = model(torch.from_numpy(input_data)).numpy()
            probabilities = softmax(outputs)[0]

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
            "name_gu": get_disease_details(CLASSES[idx]).get("name_gu", CLASSES[idx]),
            "confidence": round(float(probabilities[idx]) * 100, 2)
        }
        for idx in top3_indices
    ]

    return {
        "mode": "project",
        "status": "success",
        "is_plant": True,
        "is_confident": is_confident,
        "is_uncertain": not is_confident,
        "prediction": predicted_class,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "details": details,
        "uncertainty_note": None if is_confident else "વિશ્વાસ સ્તર ૬૦% થી ઓછું છે. આ પાન અમારા ૩૭ તાલીમબદ્ધ વર્ગોની બહારનું હોઈ શકે છે. વિગતવાર તપાસ માટે 'AI વિઝન' મોડ વાપરો.",
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
        diagnosis = analyze_plant_with_vision(image)
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
