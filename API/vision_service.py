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


# ================================================================
# ENVIRONMENT
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)


# ================================================================
# MAIN VISION PROMPT
# ================================================================

SYSTEM_PROMPT = """
You are an expert Agricultural Botanist and Plant Pathologist specialized
in Indian crops and flora.

Analyze the uploaded plant/leaf image carefully.

Provide the diagnosis strictly in Gujarati, using English scientific terms
where helpful.

IMPORTANT:
- Do not invent a disease when the visual evidence is weak.
- If uncertain, use "અનિશ્ચિત (Low)" confidence.
- Identify the plant first.
- Then identify disease, pest, nutrient deficiency, chemical damage,
  physiological problem, or Healthy.
- Give practical farmer-friendly Gujarati advice.
- Keep the answer concise.

RULES:
1. If the image is NOT a plant or leaf, return "is_plant": false.
2. If it IS a plant/leaf, identify the plant.
3. Identify the most likely disease / pest / deficiency / damage / healthy state.
4. Categorize the issue accurately.
5. Return ONLY valid raw JSON.
6. Do not wrap JSON in markdown.
7. Maximum 3 symptoms.
8. Maximum 3 management items.
9. Maximum 2 prevention items.
10. Each item must be one short sentence.

REQUIRED JSON SCHEMA:

{
  "is_plant": true,
  "plant_en": "English Plant Name",
  "plant_gu": "ગુજરાતી છોડનું નામ",
  "name_en": "English Problem Name",
  "name_gu": "ગુજરાતી રોગ / સમસ્યાનું નામ",
  "type_gu": "રોગ / જીવાત / પોષક તત્વોની ખામી / રાસાયણિક નુકસાન / સ્વસ્થ",
  "confidence_assessment": "ઉચ્ચ (High) / મધ્યમ (Medium) / અનિશ્ચિત (Low)",
  "symptoms_gu": [
    "દેખાતું લક્ષણ"
  ],
  "cause_gu": "સંભવિત કારણ",
  "management_gu": [
    "નિયંત્રણ અથવા વ્યવસ્થાપન ઉપાય"
  ],
  "prevention_gu": [
    "બચાવનું પગલું"
  ],
  "ai_note_gu": "આ પરિણામ જનરલ AI વિઝન મોડેલ દ્વારા આપેલ પ્રાથમિક વિશ્લેષણ છે."
}
"""


# ================================================================
# LEGACY PLANT CHECK
# ================================================================

PLANT_CHECK_PROMPT = """
Look at the image and decide only whether it visibly contains a real plant
or part of a plant such as a leaf, stem, fruit, or crop.

Do not infer a plant from text, packaging, a drawing, or context.

If no plant is clearly visible or you are unsure, set is_plant to false.

Return only JSON:

{"is_plant": true}

or

{"is_plant": false}
"""


NOT_PLANT_MESSAGE_GU = (
    "⚠️ આ તસવીરમાં છોડ કે પાન સ્પષ્ટ દેખાતું નથી. "
    "કૃપા કરીને છોડના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો."
)


# ================================================================
# CURRENT STABLE GEMINI MODELS
# ================================================================
#
# Order is optimized for:
# 1. quality
# 2. multimodal capability
# 3. fallback reliability
#
# The successful model is automatically moved to the front.
# ================================================================

MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


# ================================================================
# TIMEOUT / RETRY SETTINGS
# ================================================================

# Total maximum time for one Vision request.
TOTAL_DEADLINE_SECONDS = float(
    os.getenv("GEMINI_TOTAL_DEADLINE", "18")
)

# Maximum time allowed for one individual model attempt.
ATTEMPT_TIMEOUT_SECONDS = float(
    os.getenv("GEMINI_ATTEMPT_TIMEOUT", "4.5")
)

# Never start a model attempt if too little time remains.
MIN_ATTEMPT_SECONDS = 2.0


# ================================================================
# IMAGE SETTINGS
# ================================================================

VISION_MAX_SIDE = int(
    os.getenv("GEMINI_IMAGE_MAX_SIDE", "576")
)

VISION_JPEG_QUALITY = int(
    os.getenv("GEMINI_IMAGE_QUALITY", "72")
)


# ================================================================
# CACHE
# ================================================================

CACHE_MAX_ENTRIES = 128
CACHE_TTL_SECONDS = 3600


# ================================================================
# CIRCUIT BREAKER
# ================================================================

COOLDOWN_RATE_LIMIT = 30
COOLDOWN_OVERLOAD = 10
COOLDOWN_TIMEOUT = 12
COOLDOWN_NOT_FOUND = 600


# ================================================================
# SHARED HTTP SESSION
# ================================================================

_session = requests.Session()

_session.mount(
    "https://",
    HTTPAdapter(
        pool_connections=4,
        pool_maxsize=16,
        max_retries=0,
    ),
)


# ================================================================
# SHARED STATE
# ================================================================

_lock = threading.Lock()

_working_model = None

_no_thinking_models = set()

_breaker_until = {}

_cache = OrderedDict()


# ================================================================
# PER REQUEST TRACE
# ================================================================

request_trace: ContextVar = ContextVar(
    "request_trace",
    default=None
)


def _trace_add(key: str, ms: float) -> None:
    trace = request_trace.get()

    if trace is not None:
        trace[key] = trace.get(key, 0.0) + ms


def _trace_set(key: str, value) -> None:
    trace = request_trace.get()

    if trace is not None:
        trace[key] = value


# ================================================================
# CACHE
# ================================================================

def _cache_key(prompt: str, b64_data: str) -> str:

    prompt_hash = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()[:16]

    image_hash = hashlib.sha256(
        b64_data.encode("ascii")
    ).hexdigest()

    return f"{prompt_hash}:{image_hash}"


def _cache_get(key: str):

    with _lock:

        item = _cache.get(key)

        if item is None:
            return None

        timestamp, value = item

        if time.time() - timestamp > CACHE_TTL_SECONDS:

            del _cache[key]

            return None

        _cache.move_to_end(key)

        return copy.deepcopy(value)


def _cache_put(key: str, value: dict) -> None:

    with _lock:

        _cache[key] = (
            time.time(),
            copy.deepcopy(value)
        )

        _cache.move_to_end(key)

        while len(_cache) > CACHE_MAX_ENTRIES:

            _cache.popitem(last=False)


# ================================================================
# CIRCUIT BREAKER
# ================================================================

def _trip(model_name: str, seconds: float) -> None:

    with _lock:

        _breaker_until[model_name] = (
            time.monotonic() + seconds
        )


def _reset(model_name: str) -> None:

    with _lock:

        _breaker_until.pop(model_name, None)


def _ordered_candidates() -> list:

    now = time.monotonic()

    with _lock:

        until = dict(_breaker_until)

        working = _working_model

    ordered = list(MODEL_CANDIDATES)

    # Last successful model gets priority.
    if working in ordered:

        ordered.remove(working)

        ordered.insert(0, working)

    healthy = [
        model
        for model in ordered
        if until.get(model, 0) <= now
    ]

    tripped = sorted(
        [
            model
            for model in ordered
            if until.get(model, 0) > now
        ],
        key=lambda model: until[model]
    )

    return healthy + tripped


# ================================================================
# THINKING CONFIG
# ================================================================

def _thinking_config(model_name: str):

    if model_name in _no_thinking_models:
        return None

    name = model_name.lower()

    # Officially supported on 3.8 and 3.7.
    if (
        "gemini-3.8-flash" in name
        or "gemini-3.7-flash" in name
    ):
        return {
            "thinkingLevel": "low"
        }

    # Gemini 3.6 supports minimal.
    if "gemini-3.6-flash" in name:

        return {
            "thinkingLevel": "minimal"
        }

    # Gemini 3.5 supports minimal.
    if "gemini-3.5-flash" in name:

        return {
            "thinkingLevel": "minimal"
        }

    return None


# ================================================================
# IMAGE OPTIMIZATION
# ================================================================

def image_to_optimized_base64(
    image_input,
    max_side: int = VISION_MAX_SIDE,
    quality: int = VISION_JPEG_QUALITY,
) -> tuple[str, str]:

    if isinstance(image_input, (str, Path)):

        img = Image.open(image_input)

    elif isinstance(image_input, Image.Image):

        img = image_input

    else:

        raise ValueError(
            f"Invalid image input type: {type(image_input)}"
        )

    img = img.convert("RGB")

    img.thumbnail(
        (max_side, max_side),
        Image.Resampling.LANCZOS
    )

    buffered = BytesIO()

    img.save(
        buffered,
        format="JPEG",
        quality=quality,
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

    text = text.strip()

    try:

        return json.loads(text)

    except Exception:

        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:

            pass

    cleaned = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned,
        flags=re.MULTILINE
    )

    cleaned = re.sub(
        r"```$",
        "",
        cleaned
    ).strip()

    return json.loads(cleaned)


# ================================================================
# GEMINI RESPONSE TEXT
# ================================================================

def _extract_text(resp_json: dict) -> str:

    candidates = resp_json.get("candidates") or []

    if not candidates:

        raise ValueError(
            "No candidates returned from Gemini."
        )

    candidate = candidates[0]

    parts = (
        candidate.get("content") or {}
    ).get("parts") or []

    text_parts = []

    for part in parts:

        if part.get("thought"):
            continue

        text = part.get("text")

        if text:
            text_parts.append(text)

    text = "".join(text_parts)

    if not text.strip():

        reason = candidate.get(
            "finishReason",
            "unknown"
        )

        raise ValueError(
            f"Empty Gemini response "
            f"(finishReason={reason})."
        )

    return text


# ================================================================
# GEMINI REQUEST
# ================================================================

def _request_gemini(
    prompt: str,
    b64_data: str,
    mime_type: str,
    api_key: str,
    max_tokens: int | None = None,
) -> dict:

    global _working_model

    # ------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------

    cache_key = _cache_key(
        prompt,
        b64_data
    )

    cached = _cache_get(cache_key)

    if cached is not None:

        _trace_set(
            "cache",
            "hit"
        )

        _trace_set(
            "model",
            "cache"
        )

        _trace_set(
            "tries",
            0.0
        )

        return cached

    _trace_set(
        "cache",
        "miss"
    )

    # ------------------------------------------------------------
    # TOTAL DEADLINE
    # ------------------------------------------------------------

    deadline = (
        time.monotonic()
        + TOTAL_DEADLINE_SECONDS
    )

    last_error = None

    tries = 0

    models_tried = []

    # ------------------------------------------------------------
    # MODEL FALLBACK
    # ------------------------------------------------------------

    for model_name in _ordered_candidates():

        remaining = (
            deadline
            - time.monotonic()
        )

        if remaining < MIN_ATTEMPT_SECONDS:

            break

        models_tried.append(
            model_name
        )

        _trace_set(
            "models_tried",
            models_tried.copy()
        )

        # --------------------------------------------------------
        # ONE ATTEMPT PER MODEL
        # --------------------------------------------------------

        remaining = (
            deadline
            - time.monotonic()
        )

        if remaining < MIN_ATTEMPT_SECONDS:
            break

        attempt_timeout = min(
            ATTEMPT_TIMEOUT_SECONDS,
            remaining
        )

        generation_config = {
            "response_mime_type": "application/json",
        }

        if max_tokens:

            generation_config[
                "maxOutputTokens"
            ] = max_tokens

        thinking = _thinking_config(
            model_name
        )

        if thinking:

            generation_config[
                "thinkingConfig"
            ] = thinking

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
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig":
                generation_config,
        }

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model_name}:generateContent"
            f"?key={api_key}"
        )

        tries += 1

        t0 = time.perf_counter()

        try:

            response = _session.post(
                url,
                headers={
                    "Content-Type":
                        "application/json"
                },
                json=payload,
                timeout=(
                    min(5, attempt_timeout),
                    attempt_timeout,
                ),
            )

            elapsed_ms = (
                time.perf_counter()
                - t0
            ) * 1000

            _trace_add(
                "gemini",
                elapsed_ms
            )

            _trace_set(
                f"{model_name}_ms",
                round(elapsed_ms, 1)
            )

            status = response.status_code

            # ----------------------------------------------------
            # SUCCESS
            # ----------------------------------------------------

            if status == 200:

                try:

                    result = extract_json(
                        _extract_text(
                            response.json()
                        )
                    )

                except Exception as exc:

                    last_error = (
                        f"{model_name} returned "
                        f"invalid JSON: {exc}"
                    )

                    _trace_set(
                        f"{model_name}_error",
                        str(exc)
                    )

                    _trip(
                        model_name,
                        COOLDOWN_OVERLOAD
                    )

                    continue

                with _lock:

                    _working_model = model_name

                _reset(
                    model_name
                )

                _cache_put(
                    cache_key,
                    result
                )

                _trace_set(
                    "model",
                    model_name
                )

                _trace_set(
                    "tries",
                    float(tries)
                )

                _trace_set(
                    "models_tried",
                    models_tried.copy()
                )

                return result

            # ----------------------------------------------------
            # RATE LIMIT
            # ----------------------------------------------------

            if status == 429:

                last_error = (
                    f"{model_name}: "
                    "Free Tier rate limit reached."
                )

                _trace_set(
                    f"{model_name}_error",
                    last_error
                )

                _trip(
                    model_name,
                    COOLDOWN_RATE_LIMIT
                )

                # Immediately move to next model.
                continue

            # ----------------------------------------------------
            # OVERLOADED
            # ----------------------------------------------------

            if status == 503:

                last_error = (
                    f"{model_name}: "
                    "service temporarily overloaded."
                )

                _trace_set(
                    f"{model_name}_error",
                    last_error
                )

                _trip(
                    model_name,
                    COOLDOWN_OVERLOAD
                )

                continue

            # ----------------------------------------------------
            # THINKING CONFIG NOT ACCEPTED
            # ----------------------------------------------------

            if (
                status == 400
                and thinking
            ):

                _no_thinking_models.add(
                    model_name
                )

                # Retry this SAME model once
                # without thinking configuration.
                generation_config.pop(
                    "thinkingConfig",
                    None
                )

                payload[
                    "generationConfig"
                ] = generation_config

                try:

                    retry_response = _session.post(
                        url,
                        headers={
                            "Content-Type":
                                "application/json"
                        },
                        json=payload,
                        timeout=(
                            min(5, attempt_timeout),
                            attempt_timeout,
                        ),
                    )

                    retry_elapsed_ms = (
                        time.perf_counter()
                        - t0
                    ) * 1000

                    _trace_add(
                        "gemini",
                        retry_elapsed_ms
                    )

                    if (
                        retry_response.status_code
                        == 200
                    ):

                        result = extract_json(
                            _extract_text(
                        