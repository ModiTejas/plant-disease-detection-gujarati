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

def image_to_optimized_base64(image_input) -> tuple[str, str]:
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Invalid image input type: {type(image_input)}")

    img = img.convert("RGB")
    # Resize to max 768px to minimize token consumption
    img.thumbnail((768, 768), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=80, optimize=True)
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

def _request_gemini(prompt: str, b64_data: str, mime_type: str, api_key: str) -> dict:
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64_data}}
        ]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    }

    last_error = None
    for model_name in [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest"
    ]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=25
                )
                if response.status_code == 200:
                    candidates = response.json().get("candidates", [])
                    if not candidates:
                        raise ValueError("No candidates returned from Gemini.")
                    raw_text = candidates[0]["content"]["parts"][0]["text"]
                    return extract_json(raw_text)
                if response.status_code == 503:
                    time.sleep(1.0)
                    continue
                if response.status_code == 429:
                    last_error = "Free tier rate limit reached. Please wait 10 seconds before trying again."
                    break
                last_error = f"{model_name} HTTP {response.status_code}: {response.text}"
                break
            except Exception as exc:
                last_error = f"{model_name} Exception: {exc}"
                break

    raise RuntimeError(last_error or "Gemini request failed.")

def is_plant_image(image_input) -> bool:
    load_dotenv(BASE_DIR / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env file.")

    b64_data, mime_type = image_to_optimized_base64(image_input)
    result = _request_gemini(PLANT_CHECK_PROMPT, b64_data, mime_type, api_key)
    return result.get("is_plant") is True

def analyze_plant_with_vision(image_input) -> dict:
    load_dotenv(BASE_DIR / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env file.")

    if not is_plant_image(image_input):
        return {
            "is_plant": False,
            "error_gu": "⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ દેખાતું નથી. કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."
        }

    b64_data, mime_type = image_to_optimized_base64(image_input)
    return _request_gemini(SYSTEM_PROMPT, b64_data, mime_type, api_key)

if __name__ == "__main__":
    import sys
    print("Testing Vision Service with gemini-3.8-flash...")
    test_img = "Plant Disease Dataset/Tomato/Early blight/1723454500377.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    res = analyze_plant_with_vision(test_img)
    print(json.dumps(res, indent=2, ensure_ascii=False))
