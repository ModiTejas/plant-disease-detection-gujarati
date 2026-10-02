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
load_dotenv(
    BASE_DIR / ".env",
    override=True
)


# ================================================================
# SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = """
You are an expert Agricultural Botanist and Plant Pathologist
specialized in Indian crops and flora.

Analyze the uploaded image carefully and provide a practical
agricultural diagnosis.

FIRST PRIORITY — PLANT VALIDATION:

Before performing any diagnosis, determine whether a real plant
or part of a plant is clearly visible.

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

If a real plant is clearly visible, continue with the complete
diagnosis.

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

6. If the image quality is poor or the condition cannot be
   confidently identified, use the appropriate confidence level
   and clearly mention uncertainty.

7. Return ONLY a valid raw JSON object.

8. Do NOT wrap JSON inside markdown code blocks.

REQUIRED JSON SCHEMA:

{
  "is_plant": true,
  "plant_en": "English Plant Name (e.g., Mango)",
  "plant_gu": "ગુજરાતી છોડનું નામ (દા.ત. કેરી / આંબાનું ઝાડ)",
  "name_en": "English Problem Name (e.g., Anthracnose / Healthy)",
  "name_gu": "ગુજરાતી રોગ / સમસ્યાનું નામ",
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

If "is_plant" is false, return a valid JSON object using:

{
  "is_plant": false
}

You may additionally include:

"error_gu":
"⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ દેખાતું નથી. કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."
"""


# ================================================================
# IMAGE OPTIMIZATION
# ================================================================

def image_to_optimized_base64(
    image_input
) -> tuple[str, str]:

    """
    Convert an image into an optimized JPEG base64 string.

    Supported inputs:

    - File path
    - pathlib.Path
    - PIL.Image.Image

    Returns:

        (base64_image, mime_type)
    """

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------

    if isinstance(
        image_input,
        (str, Path)
    ):

        img = Image.open(
            image_input
        )

    elif isinstance(
        image_input,
        Image.Image
    ):

        img = image_input

    else:

        raise ValueError(
            f"Invalid image input type: "
            f"{type(image_input)}"
        )


    # ------------------------------------------------------------
    # Convert to RGB
    # ------------------------------------------------------------

    img = img.convert("RGB")


    # ------------------------------------------------------------
    # Resize for faster Gemini processing
    # ------------------------------------------------------------

    img.thumbnail(
        (768, 768),
        Image.Resampling.LANCZOS
    )


    # ------------------------------------------------------------
    # JPEG compression
    # ------------------------------------------------------------

    buffered = BytesIO()

    img.save(
        buffered,
        format="JPEG",
        quality=80,
        optimize=True
    )


    # ------------------------------------------------------------
    # Base64
    # ------------------------------------------------------------

    img_b64 = base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")


    return (
        img_b64,
        "image/jpeg"
    )


# ================================================================
# JSON EXTRACTION
# ================================================================

def extract_json(
    text: str
) -> dict:

    """
    Safely extract JSON from Gemini's response.

    Handles:

    - Raw JSON
    - JSON surrounded by text
    - Markdown code blocks
    """

    if not text:

        raise ValueError(
            "Gemini returned an empty response."
        )


    text = text.strip()


    # ------------------------------------------------------------
    # Attempt 1 — direct JSON
    # ------------------------------------------------------------

    try:

        result = json.loads(
            text
        )

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Gemini JSON response is not an object."
            )

        return result

    except json.JSONDecodeError:

        pass


    # ------------------------------------------------------------
    # Attempt 2 — remove markdown fences
    # ------------------------------------------------------------

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    ).strip()


    try:

        result = json.loads(
            cleaned
        )

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Gemini JSON response is not an object."
            )

        return result

    except json.JSONDecodeError:

        pass


    # ------------------------------------------------------------
    # Attempt 3 — find JSON object
    # ------------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )


    if match:

        try:

            result = json.loads(
                match.group(0)
            )

            if not isinstance(
                result,
                dict
            ):

                raise ValueError(
                    "Gemini JSON response is not an object."
                )

            return result

        except json.JSONDecodeError:

            pass


    # ------------------------------------------------------------
    # Nothing worked
    # ------------------------------------------------------------

    raise ValueError(
        "Gemini returned invalid JSON."
    )


# ================================================================
# GEMINI REQUEST
# ================================================================

def _request_gemini(
    prompt: str,
    b64_data: str,
    mime_type: str,
    api_key: str
) -> dict:

    """
    Send exactly ONE Gemini request.

    Retry policy:

    200 -> success

    429 -> STOP immediately
           No retry.

    503 -> retry once

    timeout -> retry once

    other HTTP errors -> STOP
    """


    # ============================================================
    # REQUEST PAYLOAD
    # ============================================================

    payload = {

        "contents": [

            {

                "parts": [

                    {
                        "text": prompt
                    },

                    {

                        "inline_data": {

                            "mime_type":
                                mime_type,

                            "data":
                                b64_data
                        }
                    }
                ]
            }
        ],


        "generationConfig": {

            # Force JSON
            "response_mime_type":
                "application/json",

            # Low reasoning for faster response
            "thinkingConfig": {

                "thinkingLevel":
                    "low"
            },

            # Deterministic output
            "temperature":
                0.0
        }
    }


    # ============================================================
    # MODEL
    # ============================================================

    model_name = (
        os.getenv(
            "GEMINI_MODEL"
        )
        or
        "gemini-3.8-flash"
    )


    # ============================================================
    # URL
    # ============================================================

    url = (

        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:generateContent"
        f"?key={api_key}"
    )


    last_error = None


    # ============================================================
    # TWO ATTEMPTS MAX
    #
    # Attempt 1:
    #     normal request
    #
    # Attempt 2:
    #     only 503 / timeout
    #
    # 429 NEVER retries.
    # ============================================================

    for attempt in range(2):

        try:

            response = requests.post(

                url,

                headers={
                    "Content-Type":
                        "application/json"
                },

                json=payload,

                timeout=15
            )


            # ====================================================
            # SUCCESS
            # ====================================================

            if response.status_code == 200:

                try:

                    data = response.json()

                except ValueError as exc:

                    raise RuntimeError(
                        "Gemini returned invalid HTTP JSON."
                    ) from exc


                candidates = data.get(
                    "candidates",
                    []
                )


                if not candidates:

                    raise RuntimeError(
                        "Gemini returned no candidates."
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

                    raise RuntimeError(
                        "Gemini returned no response parts."
                    )


                raw_text = parts[0].get(
                    "text",
                    ""
                )


                if not raw_text:

                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )


                return extract_json(
                    raw_text
                )


            # ====================================================
            # RATE LIMIT
            # ====================================================
            #
            # DO NOT RETRY.
            #
            # Retrying a free-tier 429 immediately can only
            # increase latency and will not solve the quota issue.
            # ====================================================

            if response.status_code == 429:

                raise RuntimeError(
                    "Free tier rate limit reached. "
                    "Please wait a few seconds before trying again."
                )


            # ====================================================
            # TEMPORARY SERVER ERROR
            # ====================================================

            if response.status_code == 503:

                last_error = (
                    "Gemini service is temporarily unavailable."
                )


                if attempt == 0:

                    # Small delay before retry
                    time.sleep(0.7)

                    continue


                break


            # ====================================================
            # OTHER HTTP ERROR
            # ====================================================

            try:

                error_data = response.json()

                error_message = (
                    error_data
                    .get("error", {})
                    .get("message")
                )

            except Exception:

                error_message = None


            last_error = (

                f"Gemini HTTP "
                f"{response.status_code}: "
                f"{error_message or response.text}"
            )


            break


        # ========================================================
        # TIMEOUT
        # ========================================================

        except requests.Timeout:

            last_error = (
                "Gemini request timed out."
            )


            if attempt == 0:

                continue


            break


        # ========================================================
        # CONNECTION ERROR
        # ========================================================

        except requests.ConnectionError as exc:

            last_error = (
                "Unable to connect to Gemini: "
                f"{exc}"
            )


            if attempt == 0:

                continue


            break


        # ========================================================
        # OTHER ERROR
        # ========================================================

        except Exception as exc:

            last_error = str(
                exc
            )

            break


    # ============================================================
    # FINAL ERROR
    # ============================================================

    raise RuntimeError(
        last_error
        or
        "Gemini request failed."
    )


# ================================================================
# MAIN PLANT VISION ANALYSIS
# ================================================================

def analyze_plant_with_vision(
    image_input
) -> dict:

    """
    Analyze a plant image using Gemini Vision.

    This function performs:

        1. Plant validation
        2. Plant identification
        3. Disease/problem identification
        4. Gujarati explanation
        5. JSON formatting

    ALL of the above happen in ONE Gemini request.
    """


    # ============================================================
    # API KEY
    # ============================================================

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


    # ============================================================
    # IMAGE OPTIMIZATION
    # ============================================================

    b64_data, mime_type = (
        image_to_optimized_base64(
            image_input
        )
    )


    # ============================================================
    # SINGLE GEMINI REQUEST
    # ============================================================

    result = _request_gemini(

        SYSTEM_PROMPT,

        b64_data,

        mime_type,

        api_key
    )


    # ============================================================
    # NON-PLANT IMAGE
    # ============================================================

    if result.get(
        "is_plant"
    ) is False:

        result.setdefault(

            "error_gu",

            "⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ "
            "દેખાતું નથી. કૃપા કરીને છોડના "
            "પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."
        )


    return result


# ================================================================
# TESTING
# ================================================================

if __name__ == "__main__":

    import sys


    print(
        "Testing Vision Service..."
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
    # Custom image
    #
    # Example:
    #
    # python vision_service.py "my_leaf.jpg"
    # ------------------------------------------------------------

    if len(sys.argv) > 1:

        test_img = sys.argv[1]


    print(
        f"\nImage: {test_img}"
    )


    print(
        "\nSending ONE request to Gemini..."
    )


    start_time = (
        time.perf_counter()
    )


    try:

        result = (
            analyze_plant_with_vision(
                test_img
            )
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
