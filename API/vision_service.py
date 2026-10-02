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


# ================================================================
# CONFIGURATION
# ================================================================

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(BASE_DIR / ".env", override=True)


# ================================================================
# SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = """
You are an expert Agricultural Botanist and Plant Pathologist specialized
in Indian crops and flora.

Analyze the uploaded image carefully and provide a practical agricultural
diagnosis.

FIRST PRIORITY — PLANT VALIDATION:

Before performing any diagnosis, determine whether a real plant or
part of a plant is clearly visible.

A plant includes:
- Leaf
- Stem
- Fruit
- Flower
- Root
- Crop
- Whole plant

If no real plant is clearly visible, return:
"is_plant": false

Do NOT infer a plant from:
- Text
- Labels
- Packaging
- Drawings
- Diagrams
- Screenshots
- Product photographs
- Context outside the visible image

If a real plant is clearly visible, continue with the complete diagnosis.

IMPORTANT:
Analyze ONLY what is visually supported by the image.
Do not invent symptoms that cannot reasonably be seen.

Provide your diagnosis strictly in Gujarati
(with English scientific terms where helpful).

RULES:

1. If the image is NOT a plant or leaf, return "is_plant": false.

2. If it IS a plant/leaf, identify:
   - Plant name
   - Disease
   - Pest
   - Nutrient deficiency
   - Physiological issue
   - Chemical damage
   - Or confirm Healthy

3. Categorize the issue accurately as one of:

   "રોગ"
   "જીવાત"
   "પોષક તત્વોની ખામી"
   "રાસાયણિક નુકસાન"
   "સ્વસ્થ"

4. Provide practical, farmer-friendly Gujarati advice.

5. Do not exaggerate certainty.

6. If the image quality is poor or the condition cannot be confidently
   identified, use the appropriate confidence level and clearly mention
   uncertainty.

7. Return ONLY a valid raw JSON object.

8. Do NOT wrap JSON inside markdown code blocks.

REQUIRED JSON SCHEMA:

{
  "is_plant": true,
  "plant_en": "English Plant Name (e.g., Mango)",
  "plant_gu": "ગુજરાતી છોડનું નામ (દા.ત. કેરી / આંબાનું ઝાડ)",
  "name_en": "English Problem Name (e.g., Anthracnose / Healthy)",
  "name_gu": "ગુજરાતી રોગ / સમસ્યાનું નામ (દા.ત. એન્થ્રેકનોઝ - કાળી મેશ રોગ)",
  "type_gu": "રોગ / જીવાત / પોષક તત્વોની ખામી / રાસાયણિક નુકસાન / સ્વસ્થ",
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


# ================================================================
# IMAGE OPTIMIZATION
# ================================================================

def image_to_optimized_base64(image_input) -> tuple[str, str]:
    """
    Convert an image into an optimized JPEG base64 string.

    Supported inputs:
    - File path
    - pathlib.Path
    - PIL.Image.Image

    Returns:
        (base64_image, mime_type)
    """

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)

    elif isinstance(image_input, Image.Image):
        img = image_input

    else:
        raise ValueError(
            f"Invalid image input type: {type(image_input)}"
        )

    # Ensure standard RGB image
    img = img.convert("RGB")

    # Keep image reasonably small for faster API processing
    img.thumbnail(
        (768, 768),
        Image.Resampling.LANCZOS
    )

    buffered = BytesIO()

    img.save(
        buffered,
        format="JPEG",
        quality=80,
        optimize=True
    )

    img_b64 = base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")

    return img_b64, "image/jpeg"


# ================================================================
# JSON EXTRACTION
# ================================================================

def extract_json(text: str) -> dict:
    """
    Safely extract JSON from Gemini's response.

    Handles:
    - Raw JSON
    - JSON surrounded by text
    - Markdown code blocks
    """

    text = text.strip()

    # ------------------------------------------------------------
    # Attempt 1: direct JSON
    # ------------------------------------------------------------

    try:
        return json.loads(text)

    except Exception:
        pass

    # ------------------------------------------------------------
    # Attempt 2: JSON object inside response
    # ------------------------------------------------------------

    match = re.search(
        r'\{.*\}',
        text,
        re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group(0))

        except Exception:
            pass

    # ------------------------------------------------------------
    # Attempt 3: remove markdown code fences
    # ------------------------------------------------------------

    cleaned = re.sub(
        r'^```json\s*',
        '',
        text,
        flags=re.MULTILINE
    )

    cleaned = re.sub(
        r'^```\s*',
        '',
        cleaned,
        flags=re.MULTILINE
    )

    cleaned = re.sub(
        r'```$',
        '',
        cleaned,
        flags=re.MULTILINE
    ).strip()

    return json.loads(cleaned)


# ================================================================
# GEMINI REQUEST
# ================================================================

def _request_gemini(
    prompt: str,
    b64_data: str,
    mime_type: str,
    api_key: str
) -> dict:

    # ------------------------------------------------------------
    # Gemini request payload
    # ------------------------------------------------------------

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    }
                ]
            }
        ],

        "generationConfig": {
            # Force structured JSON output
            "response_mime_type": "application/json",

            # Lower reasoning effort for faster response
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    # ------------------------------------------------------------
    # Primary model
    # ------------------------------------------------------------

    model_name = "gemini-3.8-flash"

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:generateContent"
        f"?key={api_key}"
    )

    last_error = None

    # ------------------------------------------------------------
    # Only two attempts:
    #
    # Attempt 1 = normal request
    # Attempt 2 = only for temporary failures/timeouts
    # ------------------------------------------------------------

    for attempt in range(2):

        try:

            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=15
            )

            # ====================================================
            # SUCCESS
            # ====================================================

            if response.status_code == 200:

                data = response.json()

                candidates = data.get(
                    "candidates",
                    []
                )

                if not candidates:
                    raise ValueError(
                        "No candidates returned from Gemini."
                    )

                content = candidates[0].get(
                    "content",
                    {}
                )

                parts = content.get(
                    "parts",
                    []
                )

                if not parts:
                    raise ValueError(
                        "Gemini returned no response parts."
                    )

                raw_text = parts[0].get(
                    "text",
                    ""
                )

                if not raw_text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                return extract_json(raw_text)

            # ====================================================
            # TEMPORARY SERVER ERROR
            # ====================================================

            if response.status_code == 503:

                last_error = (
                    "Gemini service is temporarily unavailable."
                )

                # Retry once only
                if attempt == 0:
                    time.sleep(0.7)
                    continue

                break

            # ====================================================
            # RATE LIMIT
            # ====================================================

            if response.status_code == 429:

                raise RuntimeError(
                    "Free tier rate limit reached. "
                    "Please wait a few seconds before trying again."
                )

            # ====================================================
            # OTHER HTTP ERROR
            # ====================================================

            last_error = (
                f"Gemini HTTP {response.status_code}: "
                f"{response.text}"
            )

            break

        # ========================================================
        # REQUEST TIMEOUT
        # ========================================================

        except requests.Timeout:

            last_error = (
                "Gemini request timed out."
            )

            # Retry once
            if attempt == 0:
                continue

            break

        # ========================================================
        # OTHER ERROR
        # ========================================================

        except Exception as exc:

            last_error = (
                f"Gemini Exception: {exc}"
            )

            break

    # ============================================================
    # FINAL ERROR
    # ============================================================

    raise RuntimeError(
        last_error or "Gemini request failed."
    )


# ================================================================
# MAIN PLANT VISION ANALYSIS
# ================================================================

def analyze_plant_with_vision(image_input) -> dict:
    """
    Analyze a plant image using Gemini Vision.

    IMPORTANT:
    This now performs plant validation and disease analysis
    in ONE Gemini request.

    This replaces the previous:

        is_plant_image()
            +
        analyze_plant_with_vision()

    two-request architecture.
    """

    # ------------------------------------------------------------
    # Load API key
    # ------------------------------------------------------------

    load_dotenv(
        BASE_DIR / ".env",
        override=True
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is not set in .env file."
        )

    # ------------------------------------------------------------
    # Optimize image
    # ------------------------------------------------------------

    b64_data, mime_type = (
        image_to_optimized_base64(
            image_input
        )
    )

    # ------------------------------------------------------------
    # ONE Gemini request
    #
    # Gemini performs:
    #
    # 1. Plant validation
    # 2. Plant identification
    # 3. Disease identification
    # 4. Gujarati explanation
    # 5. JSON formatting
    # ------------------------------------------------------------

    result = _request_gemini(
        SYSTEM_PROMPT,
        b64_data,
        mime_type,
        api_key
    )

    # ------------------------------------------------------------
    # Handle non-plant image
    # ------------------------------------------------------------

    if result.get("is_plant") is False:

        result.setdefault(
            "error_gu",
            "⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ દેખાતું નથી. "
            "કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."
        )

    return result


# ================================================================
# OPTIONAL LEGACY PLANT CHECK
# ================================================================
#
# Kept here for compatibility if another part of your project
# directly imports and uses is_plant_image().
#
# IMPORTANT:
# analyze_plant_with_vision() DOES NOT call this function anymore.
#
# Therefore your normal analysis uses only ONE Gemini request.
# ================================================================

def is_plant_image(image_input) -> bool:

    load_dotenv(
        BASE_DIR / ".env",
        override=True
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is not set in .env file."
        )

    plant_check_prompt = """
Look at the image and decide only whether it visibly contains
a real plant or part of a plant such as a leaf, stem, fruit,
flower, root, or crop.

Do not infer a plant from:
- text
- packaging
- labels
- drawings
- diagrams
- screenshots
- context

If no real plant is clearly visible or you are unsure,
set is_plant to false.

Return only JSON:

{"is_plant": true}

or

{"is_plant": false}
"""

    b64_data, mime_type = (
        image_to_optimized_base64(
            image_input
        )
    )

    result = _request_gemini(
        plant_check_prompt,
        b64_data,
        mime_type,
        api_key
    )

    return result.get(
        "is_plant"
    ) is True


# ================================================================
# TESTING
# ================================================================

if __name__ == "__main__":

    import sys

    print(
        "Testing Vision Service with "
        "gemini-3.8-flash..."
    )

    # ------------------------------------------------------------
    # Default test image
    # ------------------------------------------------------------

    test_img = (
        "Plant Disease Dataset/"
        "Tomato/"
        "Early blight/"
        "1723454500377.jpg"
    )

    # ------------------------------------------------------------
    # Allow custom image from command line
    #
    # Example:
    #
    # python ai_advisor.py "my_leaf.jpg"
    # ------------------------------------------------------------

    if len(sys.argv) > 1:

        test_img = sys.argv[1]

    print(
        f"\nImage: {test_img}"
    )

    print(
        "\nSending image to Gemini..."
    )

    start_time = time.perf_counter()

    try:

        result = analyze_plant_with_vision(
            test_img
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\nResponse received in "
            f"{elapsed:.2f} seconds."
        )

        print(
            "\nResult:"
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\nERROR after "
            f"{elapsed:.2f} seconds:"
        )

        print(
            str(exc)
        )
