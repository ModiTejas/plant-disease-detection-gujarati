```python
import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env", override=True)


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================
# VISION AI IMPORT
# ================================================================
#
# IMPORTANT:
#
# Project Mode (/predict)
#     -> DOES NOT use Gemini
#
# AI Vision Mode (/analyze-vision)
#     -> Uses Gemini
#
# ================================================================

try:
    from API.vision_service import analyze_plant_with_vision
except ImportError:
    from vision_service import analyze_plant_with_vision


# ================================================================
# PATHS
# ================================================================

MODEL_DIR = BASE_DIR / "models"

MODEL_ONNX = MODEL_DIR / "best_model.onnx"
MODEL_PTH = MODEL_DIR / "best_model.pth"

CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
DICT_PATH = BASE_DIR / "disease_dictionary.json"

INDEX_FILE = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"


# ================================================================
# FASTAPI
# ================================================================

app = FastAPI(
    title="Plant Disease Detection API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# CONSTANTS
# ================================================================

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60

EXPECTED_CLASSES = 37


# ================================================================
# LOAD CLASS INDICES
# ================================================================

if not CLASS_INDICES_PATH.exists():
    raise FileNotFoundError(
        f"class_indices.json not found: {CLASS_INDICES_PATH}"
    )


with open(
    CLASS_INDICES_PATH,
    "r",
    encoding="utf-8"
) as f:

    CLASS_DATA = json.load(f)


# Support both:

# {
#     "class_to_idx": {...},
#     "classes": [...]
# }

# and a direct class list if needed.

if isinstance(CLASS_DATA, dict):

    CLASSES = CLASS_DATA.get("classes")

    if not CLASSES:

        class_to_idx = CLASS_DATA.get(
            "class_to_idx",
            {}
        )

        CLASSES = [
            class_name
            for class_name, index
            in sorted(
                class_to_idx.items(),
                key=lambda item: item[1]
            )
        ]

else:

    CLASSES = CLASS_DATA


if not CLASSES:
    raise RuntimeError(
        "No classes found in class_indices.json."
    )


if len(CLASSES) != EXPECTED_CLASSES:

    raise RuntimeError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"but found {len(CLASSES)} classes."
    )


logger.info(
    "Loaded %d model classes.",
    len(CLASSES)
)


# ================================================================
# LOAD DISEASE DICTIONARY
# ================================================================

if not DICT_PATH.exists():

    raise FileNotFoundError(
        f"disease_dictionary.json not found: {DICT_PATH}"
    )


with open(
    DICT_PATH,
    "r",
    encoding="utf-8"
) as f:

    DISEASE_DICT = json.load(f)


# ================================================================
# DICTIONARY HELPERS
# ================================================================

def normalize_class_name(name: str) -> str:

    return (
        name
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# Build normalized lookup table

NORMALIZED_DICT = {
    normalize_class_name(key): value
    for key, value in DISEASE_DICT.items()
}


# Legacy aliases

CLASS_NAME_ALIASES = {

    normalize_class_name(
        "Cotton___Healthy Leaf"
    ):
        normalize_class_name(
            "Cotton___Healthy"
        ),

    normalize_class_name(
        "Cotton___Leaf Redding"
    ):
        normalize_class_name(
            "Cotton___Leaf Reddening"
        ),
}


def _as_text_list(value) -> list[str]:

    if isinstance(value, str):

        return (
            [value]
            if value.strip()
            else []
        )

    if isinstance(value, list):

        return [
            item
            for item in value
            if isinstance(item, str)
            and item.strip()
        ]

    return []


def get_disease_details(
    class_name: str
) -> dict:

    class_id = class_name.strip()

    # ------------------------------------------------------------
    # Direct lookup
    # ------------------------------------------------------------

    details = DISEASE_DICT.get(
        class_id
    )

    if details is None:

        details = NORMALIZED_DICT.get(
            normalize_class_name(class_id)
        )


    # ------------------------------------------------------------
    # Legacy alias lookup
    # ------------------------------------------------------------

    if details is None:

        normalized = normalize_class_name(
            class_id
        )

        alias = CLASS_NAME_ALIASES.get(
            normalized
        )

        if alias:

            details = NORMALIZED_DICT.get(
                alias
            )


    if details is None:

        return {}


    # ------------------------------------------------------------
    # If dictionary already has the frontend fields,
    # preserve them.
    # ------------------------------------------------------------

    sections = (
        details
        .get("farmer_result", {})
        .get("sections", {})
    )


    if not sections:

        return details


    # ------------------------------------------------------------
    # Extract farmer-friendly information
    # ------------------------------------------------------------

    chemical = sections.get(
        "chemical_control",
        {}
    )

    biological = sections.get(
        "natural_biological_control",
        {}
    )

    current_steps = sections.get(
        "what_to_do_now",
        {}
    )

    prevention = sections.get(
        "prevention",
        {}
    )


    management = (

        _as_text_list(
            chemical.get(
                "recommended_active_ingredients"
            )
        )

        +

        _as_text_list(
            chemical.get(
                "description"
            )
        )

        +

        _as_text_list(
            biological.get(
                "recommendations"
            )
        )

        +

        _as_text_list(
            current_steps.get(
                "steps"
            )
        )
    )


    return {
        **details,

        "plant_gu":
            details.get(
                "crop_name_gu",
                ""
            ),

        "name_gu":
            details.get(
                "condition_name_gu",
                ""
            ),

        "type_gu":
            details.get(
                "category_gu",
                ""
            ),

        "cause_gu":
            details.get(
                "pathogen_scientific_name",
                ""
            ),

        "symptoms_gu":
            sections.get(
                "what_happened",
                ""
            ),

        "management_gu":
            management,

        "prevention_gu":
            _as_text_list(
                prevention.get(
                    "steps"
                )
            ),
    }


# ================================================================
# INFERENCE ENGINE
# ================================================================

USE_ONNX = MODEL_ONNX.exists()

ort_session = None
model = None


if USE_ONNX:

    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        str(MODEL_ONNX),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    logger.info(
        "Inference Engine: ONNX Runtime"
    )

else:

    if not MODEL_PTH.exists():

        raise FileNotFoundError(
            "Neither ONNX nor PyTorch model was found.\n"
            f"ONNX: {MODEL_ONNX}\n"
            f"PyTorch: {MODEL_PTH}"
        )

    import torch
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v3_large(
        weights=None
    )

    in_features = (
        model.classifier[3]
        .in_features
    )

    model.classifier[3] = nn.Linear(
        in_features,
        len(CLASSES)
    )

    checkpoint = torch.load(
        MODEL_PTH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    logger.info(
        "Inference Engine: PyTorch CPU"
    )


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def preprocess_image(
    image: Image.Image
) -> np.ndarray:

    image = (
        image
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    img_arr = (
        np.array(
            image,
            dtype=np.float32
        ) / 255.0
    )

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img_arr = (
        img_arr - mean
    ) / std

    # HWC -> CHW

    img_arr = np.transpose(
        img_arr,
        (2, 0, 1)
    )

    # CHW -> NCHW

    return np.expand_dims(
        img_arr,
        axis=0
    )


# ================================================================
# SOFTMAX
# ================================================================

def softmax(x):

    e_x = np.exp(
        x - np.max(
            x,
            axis=1,
            keepdims=True
        )
    )

    return (
        e_x
        /
        e_x.sum(
            axis=1,
            keepdims=True
        )
    )


# ================================================================
# READ UPLOADED IMAGE
# ================================================================

async def read_uploaded_image(
    file: UploadFile
) -> Image.Image:

    if (
        not file.content_type
        or not file.content_type.startswith(
            "image/"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "કૃપા કરીને માન્ય ફોટો "
                "ફાઇલ અપલોડ કરો."
            )
        )


    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise ValueError(
                "Empty image file."
            )

        image = Image.open(
            BytesIO(image_bytes)
        )

        # Force actual image decoding

        image.load()

        return image.convert("RGB")

    except Exception as exc:

        logger.warning(
            "Image decoding failed: %s",
            exc
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "ફોટો ફાઇલ વાંચવામાં "
                "અસમર્થ."
            )
        ) from exc


# ================================================================
# PING
# ================================================================

@app.get("/ping")
async def ping():

    return {
        "status": "live",

        "engine":
            "ONNX"
            if USE_ONNX
            else "PyTorch",

        "total_classes":
            len(CLASSES),

        "project_mode":
            "local",

        "vision_mode":
            "gemini"
    }


# ================================================================
# PROJECT MODE
# ================================================================
#
# IMPORTANT:
#
# THERE IS NO GEMINI CALL HERE.
#
# Project Mode uses ONLY:
#
#     image
#       ↓
#     local preprocessing
#       ↓
#     ONNX / PyTorch
#       ↓
#     37-class prediction
#       ↓
#     disease_dictionary.json
#
# ================================================================

@app.post("/predict")
async def predict_project_model(
    file: UploadFile = File(...)
):

    # ------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------

    image = await read_uploaded_image(
        file
    )


    # ------------------------------------------------------------
    # Local preprocessing
    # ------------------------------------------------------------

    try:

        input_data = preprocess_image(
            image
        )

    except Exception as exc:

        logger.exception(
            "Image preprocessing failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "તસવીર તૈયાર કરવામાં "
                "સમસ્યા આવી."
            )
        ) from exc


    # ------------------------------------------------------------
    # LOCAL MODEL INFERENCE
    # ------------------------------------------------------------

    try:

        if USE_ONNX:

            outputs = (
                ort_session.run(
                    None,
                    {
                        "input":
                            input_data
                    }
                )[0]
            )

        else:

            import torch

            with torch.no_grad():

                outputs = (
                    model(
                        torch.from_numpy(
                            input_data
                        )
                    )
                    .numpy()
                )


        probabilities = softmax(
            outputs
        )[0]


    except Exception as exc:

        logger.exception(
            "Local model inference failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "સ્થાનિક AI મોડેલથી "
                "તસવીરનું વિશ્લેષણ થઈ શક્યું નથી."
            )
        ) from exc


    # ------------------------------------------------------------
    # FIND TOP PREDICTION
    # ------------------------------------------------------------

    top_idx = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[top_idx]
    )

    predicted_class = CLASSES[
        top_idx
    ]


    # ------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------

    is_confident = (
        confidence
        >= CONFIDENCE_THRESHOLD
    )

    is_uncertain = (
        not is_confident
    )


    # ------------------------------------------------------------
    # DISEASE DETAILS
    # ------------------------------------------------------------

    details = get_disease_details(
        predicted_class
    )


    # ------------------------------------------------------------
    # TOP 3 PREDICTIONS
    # ------------------------------------------------------------

    top3_indices = (
        np.argsort(
            probabilities
        )[::-1][:3]
    )


    top_3 = [

        {
            "class":
                CLASSES[idx],

            "predicted_class":
                CLASSES[idx],

            "name_gu":
                get_disease_details(
                    CLASSES[idx]
                ).get(
                    "name_gu"
                )
                or CLASSES[idx],

            "confidence":
                round(
                    float(
                        probabilities[idx]
                    ) * 100,
                    2
                )
        }

        for idx in top3_indices
    ]


    # ------------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------------

    return {

        "mode":
            "project",

        "status":
            "success",

        "is_confident":
            is_confident,

        "is_uncertain":
            is_uncertain,

        "prediction":
            predicted_class,

        "predicted_class":
            predicted_class,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "details":
            details,

        "top_3":
            top_3,

        "top_predictions":
            top_3
    }


# ================================================================
# AI VISION MODE
# ================================================================
#
# THIS is the ONLY endpoint that uses Gemini.
#
# ================================================================

@app.post("/analyze-vision")
async def predict_vision_ai(
    file: UploadFile = File(...)
):

    # ------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------

    image = await read_uploaded_image(
        file
    )


    # ------------------------------------------------------------
    # Gemini Vision
    # ------------------------------------------------------------

    try:

        diagnosis = (
            analyze_plant_with_vision(
                image
            )
        )

        return {

            "mode":
                "vision",

            "status":
                "success",

            "diagnosis":
                diagnosis
        }


    except Exception as exc:

        error_msg = str(exc)

        logger.error(
            "AI Vision failed: %s",
            error_msg
        )


        # --------------------------------------------------------
        # Rate limit
        # --------------------------------------------------------

        if (
            "429" in error_msg
            or
            "rate limit"
            in error_msg.lower()
        ):

            raise HTTPException(
                status_code=429,
                detail=(
                    "AI વિઝન હાલમાં વ્યસ્ત છે. "
                    "Gemini ની મફત ઉપયોગ મર્યાદા "
                    "પૂર્ણ થઈ ગઈ છે. "
                    "કૃપા કરીને થોડા સમય પછી "
                    "ફરી પ્રયાસ કરો."
                )
            ) from exc


        # --------------------------------------------------------
        # API key / authentication
        # --------------------------------------------------------

        if (
            "GEMINI_API_KEY"
            in error_msg
            or
            "API key"
            in error_msg
            or
            "authentication"
            in error_msg.lower()
        ):

            raise HTTPException(
                status_code=503,
                detail=(
                    "AI વિઝન સેવા ઉપલબ્ધ નથી. "
                    "કૃપા કરીને Gemini API "
                    "સેટિંગ તપાસો."
                )
            ) from exc


        # --------------------------------------------------------
        # Other Gemini error
        # --------------------------------------------------------

        raise HTTPException(
            status_code=500,
            detail=(
                "AI વિઝન વિશ્લેષણ દરમિયાન "
                "સમસ્યા આવી. કૃપા કરીને ફરી "
                "પ્રયાસ કરો."
            )
        ) from exc


# ================================================================
# FRONTEND
# ================================================================

@app.get("/")
async def serve_index():

    if not INDEX_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(
        INDEX_FILE
    )


# ================================================================
# STATIC FILES
# ================================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static"
    )


# ================================================================
# LOCAL DEVELOPMENT
# ================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "API.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
```
