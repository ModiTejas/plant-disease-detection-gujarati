import os
import re
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

SYSTEM_PROMPT = """
You are an expert Agricultural Botanist and Plant Pathologist specialized in Indian crops and flora.
Analyze the uploaded leaf/plant image carefully and diagnose its health condition.

Provide your diagnosis strictly in Gujarati (with English scientific terms where helpful) in the exact JSON format specified below.

RULES:
1. If the image is NOT a plant or leaf (e.g., human, animal, car, building), return "is_plant": false.
2. If it IS a plant/leaf, identify the plant name, the exact disease / pest / nutrient deficiency / physiological issue, or confirm if it is Healthy.
3. Categorize the issue accurately: 'રોગ' (Disease), 'જીવાત' (Pest), 'ખામી' (Deficiency), 'રાસાયણિક નુકસાન' (Damage), or 'સ્વસ્થ' (Healthy).
4. Provide practical, farmer-friendly Gujarati advice for symptoms, management, and prevention.
5. Return ONLY a valid, raw JSON object. Do not wrap in markdown code blocks.
6. Be concise: maximum 3 items in symptoms_gu, 3 in management_gu, 2 in prevention_gu.
   Each item must be one short sentence (under 15 words).

REQUIRED JSON SCHEMA:
{
  "is_plant": true,
  "plant_en": "English Plant Name (e.g., Mango)",
  "plant_gu": "ગુજરાતી છોડનું નામ (દા.ત. કેરી / આંબાનું ઝાડ)",
  "name_en": "English Problem Name (e.g., Anthracnose / Healthy)",
  "name_gu": "ગુજરાતી રોગ / સમસ્યાનું નામ (દા.ત. એન્થ્રેકનોઝ - કાળી મેશ રોગ)",
  "type_gu": "રોગ / જીવાત / પોષક તત્વોની ખામી / સ્વસ્થ",
  "confidence_assessment": "ઉચ્ચ (High) / મધ્યમ (Medium) / અનિશ્ચિત (Low)",
  "symptoms_gu": [
    "દેખાતા લક્ષણ ૧",
    "દેખાતા લક્ષણ ૨"
  ],
  "cause_gu": "સમસ્યાનું સંભવિત કારણ અથવા ફૂગ/જીવાતનું નામ",
  "management_gu": [
    "સૂચિત નિયંત્રણ / દવાનો ઉપાય ૧",
    "નિયંત્રણ ઉપાય ૨"
  ],
  "prevention_gu": [
    "ભવિષ્ય માટે બચાવના પગલાં ૧",
    "બચાવ પગલાં ૨"
  ],
  "ai_note_gu": "આ પરિણામ જનરલ AI વિઝન મોડેલ દ્વારા આપેલ પ્રાથમિક વિશ્લેષણ છે."
}
"""

PLANT_CHECK_PROMPT = """
Look at the image and decide only whether it visibly contains a real plant or part
of a plant (such as a leaf, stem, fruit, or crop). Do not infer a plant from text,
packaging, a drawing, or context. If no plant is clearly visible or you are unsure,
set is_plant to false. Return only JSON: {"is_plant": true} or {"is_plant": false}.
"""

NOT_PLANT_MESSAGE_GU = "⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ દેખાતું નથી. કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."

MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]

# ---- Speed helpers -------------------------------------------------------
# One shared HTTP session: reuses the TLS connection instead of re-handshaking.
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=16))

# Remembers the last model that worked, so later requests skip failing ones.
_working_model = None

# Models that rejected our "low thinking" setting (so we stop sending it).
_no_thinking_models = set()


def _thinking_config(model_name: str):
    """Ask the model to think as little as possible (big latency saver).
    Returns None for unknown model families; a 400 response also disables it."""
    if model_name in _no_thinking_models:
        return None
    name = model_name.lower()
    if "gemini-3" in name:
        return {"thinkingLevel": "minimal"}
    if "gemini-2.5" in name:
        return {"thinkingBudget": 0}
    return None


def image_to_optimized_base64(image_input, max_side: int = 768, quality: int = 80) -> tuple[str, str]:
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Invalid image input type: {type(image_input)}")

    img = img.convert("RGB")  # returns a copy, caller's image is not modified
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=quality, optimize=True)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64, "image/jpeg"


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    cleaned = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _extract_text(resp_json: dict) -> str:
    """Safely pull text out of a Gemini response (ignores 'thought' parts)."""
    candidates = resp_json.get("candidates") or []
    if not candidates:
        raise ValueError("No candidates returned from Gemini.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        reason = candidates[0].get("finishReason", "unknown")
        raise ValueError(f"Empty response from Gemini (finishReason={reason}).")
    return text


def _request_gemini(prompt: str, b64_data: str, mime_type: str, api_key: str,
                    max_tokens: int | None = None) -> dict:
    global _working_model

    # Try the last known-good model first, then the rest in original order.
    candidates = list(MODEL_CANDIDATES)
    if _working_model in candidates:
        candidates.remove(_working_model)
        candidates.insert(0, _working_model)

    last_error = None
    for model_name in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(2):
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            thinking = _thinking_config(model_name)
            if thinking:
                generation_config["thinkingConfig"] = thinking

            payload = {
                "contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64_data}}
                ]}],
                "generationConfig": generation_config
            }

            try:
                response = _session.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=25
                )
                if response.status_code == 200:
                    result = extract_json(_extract_text(response.json()))
                    _working_model = model_name
                    return result
                if response.status_code == 503:
                    time.sleep(0.5)
                    continue
                if response.status_code == 429:
                    last_error = "Free tier rate limit reached. Please wait 10 seconds before trying again."
                    break
                if response.status_code == 400 and thinking:
                    # This model doesn't accept our thinking setting: disable it and retry same model.
                    _no_thinking_models.add(model_name)
                    last_error = f"{model_name} HTTP 400: {response.text}"
                    continue
                last_error = f"{model_name} HTTP {response.status_code}: {response.text}"
                break
            except Exception as exc:
                last_error = f"{model_name} Exception: {exc}"
                break

    raise RuntimeError(last_error or "Gemini request failed.")


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Re-read .env in case it was edited after startup.
        load_dotenv(BASE_DIR / ".env", override=True)
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env file.")
    return api_key


def is_plant_image(image_input) -> bool:
    api_key = _get_api_key()
    # Small, low-quality image is enough to tell "plant or not" and uploads faster.
    b64_data, mime_type = image_to_optimized_base64(image_input, max_side=384, quality=70)
    result = _request_gemini(PLANT_CHECK_PROMPT, b64_data, mime_type, api_key, max_tokens=256)
    return result.get("is_plant") is True


def analyze_plant_with_vision(image_input) -> dict:
    api_key = _get_api_key()

    # Single call: SYSTEM_PROMPT already returns "is_plant": false for non-plants,
    # so the separate pre-check call is no longer needed here.
    b64_data, mime_type = image_to_optimized_base64(image_input, max_side=640, quality=75)
    result = _request_gemini(SYSTEM_PROMPT, b64_data, mime_type, api_key, max_tokens=2048)

    if result.get("is_plant") is not True:
        return {
            "is_plant": False,
            "error_gu": NOT_PLANT_MESSAGE_GU
        }
    return result


if __name__ == "__main__":
    import sys
    print("Testing Vision Service with gemini-3.8-flash...")
    test_img = "Plant Disease Dataset/Tomato/Early blight/1723454500377.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    t0 = time.time()
    res = analyze_plant_with_vision(test_img)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nTook {time.time() - t0:.2f}s")
