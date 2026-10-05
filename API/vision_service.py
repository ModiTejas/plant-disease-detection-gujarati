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

Your response MUST follow the language, script, and JSON rules below exactly.

========================
LANGUAGE & SCRIPT — VERY IMPORTANT
========================

1. ALL Gujarati explanations, descriptions, symptoms, causes, management advice,
   prevention advice, notes, categories, and plant/problem names written in Gujarati
   MUST use Gujarati script only.

2. NEVER use Tamil, Hindi/Devanagari, Bengali, Telugu, Kannada, Malayalam,
   Punjabi/Gurmukhi, Odia, or ANY other Indian-language script anywhere in a Gujarati field.

3. Do NOT mix scripts inside a Gujarati sentence.

4. English is allowed ONLY where explicitly required:
   - plant_en
   - name_en
   - scientific names inside cause_gu when scientifically necessary
   - standard English scientific terminology when it is genuinely necessary

5. When an English scientific name is used, write it ONLY in Latin/English script.
   Example:
   "Pyricularia oryzae"  ✅
   Never transliterate it into another Indian script.

6. If a scientific name needs to be written in Gujarati, use Gujarati script ONLY.
   Example:
   "પાયરિક્યુલેરિયા ઓરિઝી"  ✅
   NOT:
   "પાયரிக્યુલેરીયા ઓરિઝી"  ❌
   because the latter mixes Gujarati and Tamil scripts.

7. NEVER produce a word containing characters from another Indian script
   inside a Gujarati sentence.

8. Before returning the final JSON, perform a LANGUAGE CHECK:
   - Every Gujarati field must contain Gujarati script or permitted English scientific
     terms only.
   - There must be NO Tamil characters.
   - There must be NO Devanagari characters.
   - There must be NO Bengali characters.
   - There must be NO Telugu characters.
   - There must be NO Kannada characters.
   - There must be NO Malayalam characters.
   - There must be NO Gurmukhi characters.
   - There must be NO Odia characters.
   - If any such character appears, rewrite that word/sentence before returning JSON.

9. Do not use Hinglish.
10. Do not use unnecessary English words inside Gujarati sentences.
11. Use simple, natural, farmer-friendly Gujarati.
12. Use correct Gujarati grammar, spelling, and agricultural terminology.

========================
DIAGNOSIS RULES
========================

1. Identify:
   If the image is NOT a plant or leaf (for example human, animal, vehicle,
   object, food, building, etc.), return:
   "is_plant": false

   In this case:
   - plant_en = ""
   - plant_gu = ""
   - name_en = ""
   - name_gu = ""
   - type_gu = ""
   - confidence_assessment = ""
   - symptoms_gu = []
   - cause_gu = ""
   - management_gu = []
   - prevention_gu = []

2. Diagnose:
   If it IS a plant/leaf, identify the plant name and determine the most likely:
   - disease
   - pest
   - nutrient deficiency
   - physiological problem
   - chemical damage
   - or healthy condition

3. Do NOT invent a disease when the image does not provide enough evidence.

4. If the diagnosis is uncertain, clearly indicate:
   "અનિશ્ચિત (Low)"

5. Categorize accurately using Gujarati terms such as:
   - "રોગ"
   - "જીવાત"
   - "પોષક તત્વોની ઉણપ"
   - "રાસાયણિક નુકસાન"
   - "સ્વસ્થ"

6. Distinguish between disease, pest damage, nutrient deficiency,
   physiological disorder, and chemical damage whenever the image supports
   such a distinction.

========================
GUJARATI LANGUAGE & GRAMMAR
========================

Always write naturally understandable Gujarati.

Use:
- "પાંદડા પર કથ્થઈ રંગના ડાઘ જોવા મળે છે."
- "આ લક્ષણો ફૂગજન્ય રોગ સાથે સુસંગત છે."
- "યોગ્ય પિયત વ્યવસ્થા જાળવવી."
- "પાકની ફેરબદલી કરવી."
- "જરૂર મુજબ ફૂગનાશકનો ઉપયોગ કરવો."

Avoid:
- Hinglish
- unnecessary English
- machine-translated wording
- mixed Indian scripts
- Tamil/Devanagari transliteration
- awkward literal translations

========================
PRACTICAL AGRICULTURAL ADVICE
========================

Provide practical, farmer-friendly Gujarati advice for:
- visible symptoms
- likely cause
- management
- prevention

Use proper agricultural terminology such as:
- પિયત વ્યવસ્થા
- પાકની ફેરબદલી
- ફૂગનાશક
- જીવાતનાશક
- જૈવિક નિયંત્રણ
- સંક્રમિત પાંદડા દૂર કરવા
- યોગ્ય નિકાલ
- સંતુલિત ખાતર વ્યવસ્થા

Do not provide an exact pesticide dose unless you are sufficiently confident
that the recommendation is appropriate for the identified crop and problem.
When uncertain, give safer general management advice and recommend consultation
with a local agricultural expert.

========================
OUTPUT FORMAT
========================

Return ONLY a valid, raw JSON object.

DO NOT:
- wrap JSON in markdown
- use ```json
- add introductory text
- add concluding text
- add explanations outside the JSON
- add comments inside JSON

The JSON must exactly follow this schema:

{
  "is_plant": true,
  "plant_en": "English Plant Name (e.g., Mango)",
  "plant_gu": "ગુજરાતી છોડનું નામ (દા.ત. આંબાનું ઝાડ / કેરી)",
  "name_en": "English Problem Name (e.g., Anthracnose / Healthy)",
  "name_gu": "ગુજરાતી રોગ / સમસ્યાનું નામ (દા.ત. એન્થ્રેકનોઝ - કાળો ડાઘ / કાળી મેશ રોગ)",
  "type_gu": "રોગ / જીવાત / પોષક તત્વોની ઉણપ / રાસાયણિક નુકસાન / સ્વસ્થ",
  "confidence_assessment": "ઉચ્ચ (High) / મધ્યમ (Medium) / અનિશ્ચિત (Low)",
  "symptoms_gu": [
    "દેખાતા લક્ષણ ૧ (દા.ત. પાંદડા પર કથ્થઈ રંગના ડાઘ જોવા મળે છે)",
    "દેખાતા લક્ષણ ૨"
  ],
  "cause_gu": "સમસ્યાનું સંભવિત કારણ અથવા ફૂગ/જીવાતનું વૈજ્ઞાનિક નામ",
  "management_gu": [
    "રાસાયણિક અથવા જૈવિક નિયંત્રણ ઉપાય ૧",
    "ઉપાય ૨"
  ],
  "prevention_gu": [
    "ભવિષ્યમાં રોગ અટકાવવા માટેનું આગોતરું પગલું ૧",
    "આગોતરું પગલું ૨"
  ],
  "ai_note_gu": "આ પરિણામ જનરલ AI વિઝન મોડેલ દ્વારા આપવામાં આવેલ પ્રાથમિક વિશ્લેષણ છે. કોઈપણ દવાનો છંટકાવ કરતા પહેલા ખેડૂતમિત્રોએ સ્થાનિક કૃષિ નિષ્ણાત અથવા ગ્રામસેવકની સલાહ લેવી હિતાવહ છે."
}

========================
FINAL SELF-CHECK — MANDATORY
========================

Before returning the JSON, silently verify ALL of the following:

[ ] The response is valid JSON.
[ ] No markdown code block is used.
[ ] No text exists outside the JSON.
[ ] Gujarati fields are written in natural Gujarati.
[ ] No Tamil script appears anywhere in Gujarati text.
[ ] No Devanagari script appears anywhere in Gujarati text.
[ ] No Bengali script appears anywhere.
[ ] No Telugu script appears anywhere.
[ ] No Kannada script appears anywhere.
[ ] No Malayalam script appears anywhere.
[ ] No Gurmukhi script appears anywhere.
[ ] No Odia script appears anywhere.
[ ] English is used only where permitted.
[ ] Scientific names are either correctly written in English/Latin script
    or correctly transliterated into Gujarati script.
[ ] No Gujarati sentence contains mixed Indian scripts.
[ ] The diagnosis is supported by the visible image.
[ ] Uncertainty is explicitly stated when appropriate.

If any check fails, correct the response BEFORE returning it.
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
        "gemini-3.5-flash-lite",    # fastest, most available on free tier
        "gemini-3.8-flash",          # newest, stable
        "gemini-3.6-flash",          # mid-tier backup
        "gemini-2.5-flash",          # stable backup
        "gemini-3.5-flash",          # heavier, last resort
    ]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=12
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
