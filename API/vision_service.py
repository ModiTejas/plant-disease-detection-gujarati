import os
import re
import json
import time
import copy
import random
import base64
import hashlib
import threading
from collections import OrderedDict
from contextvars import ContextVar
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

# ---- Tunables (can be overridden with env vars, no code change needed) ----
TOTAL_DEADLINE_SECONDS = float(os.getenv("GEMINI_TOTAL_DEADLINE", "20"))   # hard budget per Gemini request
ATTEMPT_TIMEOUT_SECONDS = float(os.getenv("GEMINI_ATTEMPT_TIMEOUT", "15"))  # max for a single HTTP attempt
MIN_ATTEMPT_SECONDS = 2.0     # don't start an attempt with less time than this left
CACHE_MAX_ENTRIES = 128
CACHE_TTL_SECONDS = 3600

# Circuit-breaker cool-downs (seconds) per failure type
COOLDOWN_RATE_LIMIT = 30
COOLDOWN_OVERLOAD = 15
COOLDOWN_TIMEOUT = 20
COOLDOWN_NOT_FOUND = 600

# ---- Shared state --------------------------------------------------------
# One shared HTTP session: reuses the TLS connection instead of re-handshaking.
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=16))

_lock = threading.Lock()
_working_model = None              # last model that answered successfully
_no_thinking_models = set()        # models that rejected our "low thinking" setting
_breaker_until = {}                # model -> time.monotonic() until which it is skipped
_cache = OrderedDict()             # key -> (timestamp, result)

# Per-request timing trace. main.py sets a dict here; we fill it in.
request_trace: ContextVar = ContextVar("request_trace", default=None)


def _trace_add(key: str, ms: float) -> None:
    trace = request_trace.get()
    if trace is not None:
        trace[key] = trace.get(key, 0.0) + ms


def _trace_set(key: str, value) -> None:
    trace = request_trace.get()
    if trace is not None:
        trace[key] = value


# ---- Cache (identical image + prompt => instant answer, no Gemini call) ---
def _cache_key(prompt: str, b64_data: str) -> str:
    p = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    d = hashlib.sha256(b64_data.encode("ascii")).hexdigest()
    return f"{p}:{d}"


def _cache_get(key: str):
    with _lock:
        item = _cache.get(key)
        if item is None:
            return None
        ts, value = item
        if time.time() - ts > CACHE_TTL_SECONDS:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return copy.deepcopy(value)


def _cache_put(key: str, value: dict) -> None:
    with _lock:
        _cache[key] = (time.time(), copy.deepcopy(value))
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


# ---- Circuit breaker (skip a failing model for a short cool-down) ---------
def _trip(model_name: str, seconds: float) -> None:
    with _lock:
        _breaker_until[model_name] = time.monotonic() + seconds


def _reset(model_name: str) -> None:
    with _lock:
        _breaker_until.pop(model_name, None)


def _ordered_candidates() -> list:
    """Healthy models first (last working one at the very front);
    tripped models go last (soonest to recover first) so we never fail
    purely because of the breaker."""
    now = time.monotonic()
    with _lock:
        until = dict(_breaker_until)
        working = _working_model

    ordered = list(MODEL_CANDIDATES)
    if working in ordered:
        ordered.remove(working)
        ordered.insert(0, working)

    healthy = [m for m in ordered if until.get(m, 0) <= now]
    tripped = sorted((m for m in ordered if until.get(m, 0) > now), key=lambda m: until[m])
    return healthy + tripped


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

    # 1) Cache: same image + same prompt seen recently => no network call at all.
    cache_key = _cache_key(prompt, b64_data)
    cached = _cache_get(cache_key)
    if cached is not None:
        _trace_set("cache", "hit")
        return cached
    _trace_set("cache", "miss")

    # 2) One total time budget shared by every model/attempt.
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS
    last_error = None
    tries = 0

    for model_name in _ordered_candidates():
        if deadline - time.monotonic() < MIN_ATTEMPT_SECONDS:
            break

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        cooldown = None  # set when this model fails; trips the breaker afterwards

        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining < MIN_ATTEMPT_SECONDS:
                break

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

            tries += 1
            t0 = time.perf_counter()
            try:
                response = _session.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=(5, min(ATTEMPT_TIMEOUT_SECONDS, remaining))
                )
                _trace_add("gemini", (time.perf_counter() - t0) * 1000)

                if response.status_code == 200:
                    result = extract_json(_extract_text(response.json()))
                    with _lock:
                        _working_model = model_name
                    _reset(model_name)
                    _cache_put(cache_key, result)
                    _trace_set("model", model_name)
                    _trace_set("tries", float(tries))
                    return result

                if response.status_code == 503:
                    cooldown = COOLDOWN_OVERLOAD
                    last_error = f"{model_name} HTTP 503: model overloaded"
                    time.sleep(random.uniform(0.3, 0.7))  # jitter
                    continue
                if response.status_code == 429:
                    cooldown = COOLDOWN_RATE_LIMIT
                    last_error = "Free tier rate limit reached. Please wait 10 seconds before trying again."
                    break
                if response.status_code == 400 and thinking:
                    # Model doesn't accept our thinking setting: disable it and retry same model.
                    _no_thinking_models.add(model_name)
                    last_error = f"{model_name} HTTP 400: {response.text}"
                    continue
                if response.status_code == 404:
                    cooldown = COOLDOWN_NOT_FOUND
                    last_error = f"{model_name} HTTP 404: {response.text}"
                    break
                if response.status_code >= 500:
                    cooldown = COOLDOWN_OVERLOAD
                last_error = f"{model_name} HTTP {response.status_code}: {response.text}"
                break

            except requests.exceptions.Timeout:
                _trace_add("gemini", (time.perf_counter() - t0) * 1000)
                cooldown = COOLDOWN_TIMEOUT
                last_error = f"{model_name} timeout after {time.perf_counter() - t0:.1f}s"
                break
            except Exception as exc:
                _trace_add("gemini", (time.perf_counter() - t0) * 1000)
                last_error = f"{model_name} Exception: {exc}"
                break

        if cooldown:
            _trip(model_name, cooldown)

    _trace_set("tries", float(tries))
    raise RuntimeError(last_error or "Gemini request timed out (time budget exhausted).")


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
    t0 = time.perf_counter()
    b64_data, mime_type = image_to_optimized_base64(image_input, max_side=384, quality=70)
    _trace_add("encode", (time.perf_counter() - t0) * 1000)
    result = _request_gemini(PLANT_CHECK_PROMPT, b64_data, mime_type, api_key, max_tokens=256)
    return result.get("is_plant") is True


def analyze_plant_with_vision(image_input) -> dict:
    api_key = _get_api_key()

    # Single call: SYSTEM_PROMPT already returns "is_plant": false for non-plants,
    # so the separate pre-check call is not needed here.
    t0 = time.perf_counter()
    b64_data, mime_type = image_to_optimized_base64(image_input, max_side=640, quality=75)
    _trace_add("encode", (time.perf_counter() - t0) * 1000)
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
    trace = {}
    request_trace.set(trace)
    t0 = time.time()
    res = analyze_plant_with_vision(test_img)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nTook {time.time() - t0:.2f}s | trace: {trace}")
