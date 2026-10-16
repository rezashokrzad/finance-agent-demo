"""Model adapters for JEV, Ollama, DeepSeek, and Claude.

Credentials come from the environment. This module does not invent token
counts, prices, or successful responses.
"""

import json
import os
import re
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
MAX_PROMPT_CHARS = 8000
CATEGORIES = ("invoice", "expense", "financial_report", "other")
EXTRACTION_KEYS = (
    "document_type",
    "date",
    "vendor",
    "amount",
    "currency",
    "tax",
    "description",
)
ANALYSIS_KEYS = (
    "insights",
    "anomalies",
    "missing_information",
    "risks",
    "supported_by_document",
    "assumptions",
)
VALIDATION_LIST_KEYS = (
    "arithmetic_notes",
    "consistency_notes",
    "unsupported_claims",
    "missing_fields",
)

EXTRACT_SYSTEM = (
    "You extract financial fields from one document for a classroom demo. "
    "Reply with one JSON object only, with no markdown. "
    "Required keys: document_type, date, vendor, amount, currency, tax, description. "
    "Use JSON null when a value is not in the document. "
    "amount and tax must be numbers when they are present, without currency symbols. "
    "Do not invent values."
)

EXTRACT_HINTS = {
    "invoice": (
        "This document was routed as an invoice. "
        "Prefer the invoice date, the supplier as vendor, the total amount due as amount, "
        "and the tax amount as tax."
    ),
    "expense": (
        "This document was routed as an expense. "
        "Prefer the expense date, the merchant as vendor, the total amount as amount, "
        "and the tax amount as tax."
    ),
    "financial_report": (
        "This document was routed as a financial report. "
        "Use the reporting organization as vendor, the period end as date, "
        "net income as amount, and income tax as tax. "
        "Put the reporting period in description."
    ),
    "other": (
        "This document was routed as other. "
        "Extract only financial fields you can see. Use null for anything absent."
    ),
}

ANALYZE_SYSTEM = (
    "You analyze extracted financial fields for a classroom demo. "
    "Reply with one JSON object only, with no markdown. "
    "Each of these keys must be a list of strings: "
    "insights, anomalies, missing_information, risks, supported_by_document, assumptions. "
    "Put statements the source text supports in supported_by_document. "
    "Put anything you inferred in assumptions. "
    "Do not invent amounts that are not in the input. "
    "Do not claim the books have been audited."
)

VALIDATE_SYSTEM = (
    "You are an advisory checker for a classroom finance demo. "
    "You are not proof that the figures are financially correct. "
    "Compare the source text, the extracted JSON, and the analysis. "
    "Check arithmetic, internal consistency, unsupported claims, and missing fields. "
    "Reply with one JSON object only, with no markdown. "
    'Use key "status" with one of pass, fail, review. '
    'Use key "explanation" with a short string. '
    "Use these list-of-string keys: arithmetic_notes, consistency_notes, "
    "unsupported_claims, missing_fields. "
    "Use fail when figures contradict the source. "
    "Use review when the evidence is incomplete. "
    "Use pass only when the extraction and analysis agree with the source "
    "and the arithmetic you can check."
)


class ModelError(Exception):
    def __init__(self, message, latency_s=None):
        super().__init__(message)
        self.latency_s = latency_s


class ConfigError(ModelError):
    """The .env file is missing a value or has an unusable value."""


@dataclass(frozen=True)
class Settings:
    jev_api_key: str
    jev_model: str
    jev_input_cost: str
    jev_output_cost: str
    model_1_api_key: str
    model_1_model: str
    model_1_input_cost: str
    model_1_output_cost: str
    model_2_api_key: str
    model_2_model: str
    model_2_input_cost: str
    model_2_output_cost: str
    ollama_base_url: str
    ollama_model: str
    timeout_seconds: float

    def secrets(self):
        return (self.jev_api_key, self.model_1_api_key, self.model_2_api_key)


def _text(env, key):
    value = env.get(key)
    if value is None:
        return ""
    return str(value).strip()


def settings_from_mapping(env):
    timeout_raw = _text(env, "REQUEST_TIMEOUT_SECONDS") or "180"
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError("REQUEST_TIMEOUT_SECONDS must be a number of seconds.") from exc
    if timeout <= 0:
        raise ConfigError("REQUEST_TIMEOUT_SECONDS must be greater than 0.")
    return Settings(
        jev_api_key=_text(env, "JEV_API_KEY"),
        jev_model=_text(env, "JEV_MODEL") or "jev-latest",
        jev_input_cost=_text(env, "JEV_INPUT_USD_PER_MILLION"),
        jev_output_cost=_text(env, "JEV_OUTPUT_USD_PER_MILLION"),
        model_1_api_key=_text(env, "MODEL_1_API_KEY"),
        model_1_model=_text(env, "MODEL_1_MODEL") or "deepseek-flash",
        model_1_input_cost=_text(env, "MODEL_1_INPUT_USD_PER_MILLION"),
        model_1_output_cost=_text(env, "MODEL_1_OUTPUT_USD_PER_MILLION"),
        model_2_api_key=_text(env, "MODEL_2_API_KEY"),
        model_2_model=_text(env, "MODEL_2_MODEL") or "claude-sonnet-5-5",
        model_2_input_cost=_text(env, "MODEL_2_INPUT_USD_PER_MILLION"),
        model_2_output_cost=_text(env, "MODEL_2_OUTPUT_USD_PER_MILLION"),
        ollama_base_url=_text(env, "OLLAMA_BASE_URL") or "http://127.0.0.1:11434",
        ollama_model=_text(env, "OLLAMA_MODEL") or "qwen2.5:0.5b",
        timeout_seconds=timeout,
    )


def load_settings():
    load_dotenv(ENV_PATH, override=True)
    return settings_from_mapping(os.environ)


def public_config_line(settings):
    """Status text that is safe to show in the page. It never includes key values."""

    def mark(value):
        return "set" if str(value).strip() else "not set"

    def model_name(value):
        return value.strip() if str(value).strip() else "not set"

    return (
        f"JEV: key {mark(settings.jev_api_key)}, "
        f"model {model_name(settings.jev_model)}. "
        f"API Model 1 (DeepSeek): key {mark(settings.model_1_api_key)}, "
        f"model {model_name(settings.model_1_model)}. "
        f"API Model 2 (Claude): key {mark(settings.model_2_api_key)}, "
        f"model {model_name(settings.model_2_model)}. "
        f"Ollama model: {settings.ollama_model}. "
        f"Ollama URL: {settings.ollama_base_url}. "
        "API keys are not shown."
    )


def scrub(text, settings):
    cleaned = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", str(text), flags=re.IGNORECASE)
    for secret in settings.secrets():
        if secret and len(secret) >= 8:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def require_configured(value, label, env_name):
    if not str(value).strip():
        raise ConfigError(
            f"{label} is not set. Add {env_name} to .env and run the workflow again."
        )


def clip_text(text, limit=MAX_PROMPT_CHARS):
    if len(text) <= limit:
        return text
    return text[: limit] + "\n[truncated]"


def snippet(text, limit=180):
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def blank_metric():
    return {
        "status": "not_called",
        "model": None,
        "latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
    }


def blank_metrics():
    return {
        "jev": blank_metric(),
        "ollama": blank_metric(),
        "model_1": blank_metric(),
        "model_2": blank_metric(),
    }


def build_metric(status, model, latency_s, usage, cost):
    usage = usage or {}
    return {
        "status": status,
        "model": model,
        "latency_s": latency_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_cost_usd": cost,
    }


def failed_metric(model, latency_s):
    return build_metric("failed", model, latency_s, None, None)


def first_int(payload, *keys):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return int(value)
    return None


def pack_usage(prompt, completion, total):
    usage = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is not None:
        usage["total_tokens"] = total
    return usage or None


def usage_from_openai(data):
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    return pack_usage(
        first_int(usage, "prompt_tokens", "input_tokens"),
        first_int(usage, "completion_tokens", "output_tokens"),
        first_int(usage, "total_tokens"),
    )


def usage_from_ollama(data):
    if not isinstance(data, dict):
        return None
    return pack_usage(
        first_int(data, "prompt_eval_count"),
        first_int(data, "eval_count"),
        None,
    )


def estimate_cost(usage, input_raw, output_raw, label):
    """Return (cost, error). Missing prices or missing usage produce no invented cost."""
    input_raw = (input_raw or "").strip()
    output_raw = (output_raw or "").strip()
    if input_raw == "" and output_raw == "":
        return None, None
    if input_raw == "" or output_raw == "":
        return None, (
            f"{label}: set both the input and output USD-per-million prices, or leave both blank. "
            "Estimated cost was not calculated."
        )
    try:
        input_price = float(input_raw)
        output_price = float(output_raw)
    except ValueError:
        return None, (
            f"{label}: token prices must be numbers (USD per 1,000,000 tokens). "
            "Estimated cost was not calculated."
        )
    if input_price < 0 or output_price < 0:
        return None, f"{label}: token prices cannot be negative. Estimated cost was not calculated."
    if not usage:
        return None, None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return None, None
    cost = (prompt * input_price + completion * output_price) / 1_000_000
    return cost, None


def parse_json_object(content):
    if not isinstance(content, str) or not content.strip():
        raise ModelError("The model returned an empty response.")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ModelError(
                "The model response was not valid JSON. Response started with: " + snippet(text)
            )
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ModelError(
                f"The model response was not valid JSON ({exc}). Response started with: {snippet(text)}"
            ) from exc
    if not isinstance(value, dict):
        raise ModelError("The model JSON must be an object.")
    return value


def optional_confidence(value):
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if 0 <= number <= 1:
        return number
    return None


def parse_classification(payload):
    raw = payload.get("category")
    category = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if category not in CATEGORIES:
        raise ModelError(f"JEV returned an invalid category: {raw!r}.")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = None
    else:
        reason = reason.strip()
    return {
        "category": category,
        "confidence": optional_confidence(payload.get("confidence")),
        "reason": reason,
    }


def clean_number(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text == "" or text.lower() in {"null", "none", "n/a", "na"}:
            return None
        try:
            return float(text)
        except ValueError:
            return value.strip()
    return value


def clean_field(key, value):
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"null", "none", "n/a", "na"}:
            return None
    if value is None:
        return None
    if key in {"amount", "tax"}:
        return clean_number(value)
    return value


def parse_extraction(payload):
    missing = [key for key in EXTRACTION_KEYS if key not in payload]
    if missing:
        raise ModelError(
            "The local model JSON is missing required fields: "
            + ", ".join(missing)
            + ". The workflow did not substitute its own extraction."
        )
    ordered = {key: clean_field(key, payload[key]) for key in EXTRACTION_KEYS}
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def parse_string_lists(payload, keys, model_label):
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ModelError(
            f"{model_label} JSON is missing required fields: " + ", ".join(missing) + "."
        )
    ordered = {}
    for key in keys:
        value = payload[key]
        if not isinstance(value, list):
            raise ModelError(f"{model_label} field {key} must be a list.")
        ordered[key] = value
    return ordered


def parse_analysis(payload):
    return parse_string_lists(payload, ANALYSIS_KEYS, "DeepSeek")


def parse_validation(payload):
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"pass", "fail", "review"}:
        raise ModelError(f"Claude returned an invalid status: {payload.get('status')!r}.")
    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ModelError("Claude JSON is missing a text explanation.")
    lists = parse_string_lists(payload, VALIDATION_LIST_KEYS, "Claude")
    return {"status": status, "explanation": explanation.strip(), **lists}


def message_text(content):
    if isinstance(content, str):
        if not content.strip():
            raise ModelError("The model returned an empty response.")
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "\n".join(part for part in parts if part.strip())
        if joined.strip():
            return joined
    raise ModelError("The model response did not include text content.")


def response_detail(response):
    try:
        payload = response.json()
    except ValueError:
        return " ".join(response.text.split())[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        if error is not None:
            return str(error)[:300]
        if payload.get("message"):
            return str(payload["message"])[:300]
    return " ".join(response.text.split())[:300]


def request_json(url, payload, headers, settings, connection_message, read_message):
    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(10, settings.timeout_seconds),
        )
    except requests.ConnectTimeout as exc:
        latency = time.perf_counter() - started
        raise ModelError(scrub(connection_message, settings), latency_s=latency) from exc
    except requests.ReadTimeout as exc:
        latency = time.perf_counter() - started
        raise ModelError(scrub(read_message, settings), latency_s=latency) from exc
    except requests.ConnectionError as exc:
        latency = time.perf_counter() - started
        raise ModelError(scrub(connection_message, settings), latency_s=latency) from exc
    except requests.RequestException as exc:
        latency = time.perf_counter() - started
        raise ModelError(scrub(f"The request to {url} failed.", settings), latency_s=latency) from exc
    return response, time.perf_counter() - started


def read_json_body(response, settings, latency):
    try:
        data = response.json()
    except ValueError as exc:
        raise ModelError(
            "The provider returned a non-JSON response: " + snippet(scrub(response.text, settings)),
            latency_s=latency,
        ) from exc
    if not isinstance(data, dict):
        raise ModelError("The provider returned JSON that was not an object.", latency_s=latency)
    return data


def openai_chat(base, key, model, messages, settings, label, key_env, model_env, base_env=None, extra=None):
    require_configured(key, f"{label} API key", key_env)
    if base_env:
        require_configured(base, f"{label} API endpoint", base_env)
    elif not str(base).strip():
        raise ConfigError(f"{label} API endpoint is not set.")
    require_configured(model, f"{label} model name", model_env)
    url = base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0, "messages": messages}
    if extra:
        payload.update(extra)
    connection_message = f"Could not connect to {label} at {url}."
    read_message = (
        f"{label} did not respond within {settings.timeout_seconds:g} seconds. "
        "The workflow did not substitute another model."
    )
    response, latency = request_json(
        url,
        payload,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        settings,
        connection_message,
        read_message,
    )
    if response.status_code >= 400:
        detail = scrub(response_detail(response), settings)
        raise ModelError(
            f"{label} returned HTTP {response.status_code}: {detail}",
            latency_s=latency,
        )
    data = read_json_body(response, settings, latency)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(
            f"{label} response did not include message content.",
            latency_s=latency,
        ) from exc
    return message_text(content), latency, usage_from_openai(data)


def claude_text(content):
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "\n".join(part for part in parts if part.strip()).strip()
        if joined:
            return joined
    raise ModelError("Claude response did not include text content.")


def claude_messages(key, model, system, user, settings):
    require_configured(key, "Claude API key", "MODEL_2_API_KEY")
    require_configured(model, "Claude model name", "MODEL_2_MODEL")
    url = "https://api.anthropic.com/v1/messages"
    connection_message = f"Could not connect to Claude at {url}."
    read_message = (
        f"Claude did not respond within {settings.timeout_seconds:g} seconds. "
        "The workflow did not substitute another model."
    )
    response, latency = request_json(
        url,
        {
            "model": model,
            "max_tokens": 4096,
            "thinking": {"type": "between_tools"},
            "output_config": {"effort": "low"},
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        settings,
        connection_message,
        read_message,
    )
    if response.status_code >= 400:
        detail = scrub(response_detail(response), settings)
        raise ModelError(
            f"Claude returned HTTP {response.status_code}: {detail}",
            latency_s=latency,
        )
    data = read_json_body(response, settings, latency)
    try:
        content = claude_text(data.get("content"))
    except ModelError as exc:
        raise ModelError(str(exc), latency_s=latency) from exc
    return content, latency, usage_from_openai(data)


def ollama_chat(messages, settings):
    require_configured(settings.ollama_base_url, "Ollama base URL", "OLLAMA_BASE_URL")
    require_configured(settings.ollama_model, "Ollama model name", "OLLAMA_MODEL")
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    model = settings.ollama_model
    connection_message = (
        f"Could not connect to Ollama at {url}. "
        "Start the Ollama application, then install the configured model with: "
        f"ollama pull {model}"
    )
    read_message = (
        f"Ollama did not finish within {settings.timeout_seconds:g} seconds. "
        "The local model was not replaced. Raise REQUEST_TIMEOUT_SECONDS or run the workflow again."
    )
    response, latency = request_json(
        url,
        {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": messages,
            "options": {"temperature": 0},
        },
        {"Content-Type": "application/json"},
        settings,
        connection_message,
        read_message,
    )
    if response.status_code >= 400:
        detail = scrub(response_detail(response), settings)
        missing_model = response.status_code == 404 or "not found" in detail.lower()
        if missing_model:
            extra = f" Server said: {detail}" if detail else ""
            raise ModelError(
                f"Ollama model {model!r} is not available at {url}. "
                f"Install it with: ollama pull {model}.{extra}",
                latency_s=latency,
            )
        raise ModelError(
            f"Ollama returned HTTP {response.status_code}: {detail}",
            latency_s=latency,
        )
    data = read_json_body(response, settings, latency)
    try:
        content = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ModelError("The Ollama response did not include message content.", latency_s=latency) from exc
    return message_text(content), latency, usage_from_ollama(data)


def dump_json(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def finish_call(content, parser, latency, usage, model_name, input_cost, output_cost, label):
    parsed = parser(parse_json_object(content))
    cost, cost_error = estimate_cost(usage, input_cost, output_cost, label)
    return parsed, build_metric("completed", model_name, latency, usage, cost), cost_error


def jev_reason(choice, probabilities):
    selected = str(choice or "").strip() or "no category"
    if not isinstance(probabilities, dict):
        return f"JEV selected {selected}."
    parts = []
    for name in CATEGORIES:
        value = probabilities.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parts.append(f"{name} {float(value):.2f}")
    if not parts:
        return f"JEV selected {selected}."
    return f"JEV selected {selected}. Probabilities: " + ", ".join(parts) + "."


def classify_with_jev(text, settings):
    require_configured(settings.jev_api_key, "JEV API key", "JEV_API_KEY")
    require_configured(settings.jev_model, "JEV model name", "JEV_MODEL")
    url = "https://api.typesafe.ai/v1/systemone"
    response, latency = request_json(
        url,
        {
            "model": settings.jev_model,
            "state": clip_text(text),
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Which processing path should this business document take?",
                    "criteria": {
                        "invoice": "A bill from a vendor requesting payment for goods or services.",
                        "expense": "A record of money already spent, such as a receipt or an expense line.",
                        "financial_report": "A statement of financial results for a period, such as revenue, expenses, or profit.",
                        "other": "A document that is not an invoice, an expense, or a financial report.",
                    },
                }
            },
        },
        {
            "Authorization": f"Bearer {settings.jev_api_key}",
            "Content-Type": "application/json",
        },
        settings,
        f"Could not connect to JEV at {url}.",
        (
            f"JEV did not respond within {settings.timeout_seconds:g} seconds. "
            "The workflow did not substitute another model."
        ),
    )
    if response.status_code >= 400:
        detail = scrub(response_detail(response), settings)
        raise ModelError(
            f"JEV returned HTTP {response.status_code}: {detail}",
            latency_s=latency,
        )
    data = read_json_body(response, settings, latency)
    answers = data.get("answers") if isinstance(data, dict) else None
    answer = answers.get("category") if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise ModelError("JEV response did not include a category answer.", latency_s=latency)
    try:
        parsed = parse_classification(
            {
                "category": answer.get("choice"),
                "confidence": answer.get("confidence"),
                "reason": jev_reason(answer.get("choice"), answer.get("probabilities")),
            }
        )
    except ModelError as exc:
        raise ModelError(str(exc), latency_s=latency) from exc
    usage = usage_from_openai(data)
    returned_model = data.get("model")
    if isinstance(returned_model, str) and returned_model.strip():
        model_name = returned_model.strip()
    else:
        model_name = settings.jev_model
    cost, cost_error = estimate_cost(usage, settings.jev_input_cost, settings.jev_output_cost, "JEV")
    return parsed, build_metric("completed", model_name, latency, usage, cost), cost_error


def extract_with_ollama(text, category, settings):
    hint = EXTRACT_HINTS.get(category, EXTRACT_HINTS["other"])
    content, latency, usage = ollama_chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": hint + "\n\nDocument:\n" + clip_text(text)},
        ],
        settings,
    )
    # Local inference has no API price in this demo.
    parsed = parse_extraction(parse_json_object(content))
    return parsed, build_metric("completed", settings.ollama_model, latency, usage, None), None


def analyze_with_model1(text, category, extraction, settings):
    user = (
        f"Route: {category}\n\n"
        "Extracted JSON:\n"
        f"{dump_json(extraction)}\n\n"
        "Document:\n"
        f"{clip_text(text)}"
    )
    content, latency, usage = openai_chat(
        "https://api.deepseek.com",
        settings.model_1_api_key,
        settings.model_1_model,
        [
            {"role": "system", "content": ANALYZE_SYSTEM},
            {"role": "user", "content": user},
        ],
        settings,
        "DeepSeek",
        "MODEL_1_API_KEY",
        "MODEL_1_MODEL",
        extra={"thinking": {"type": "disabled"}},
    )
    return finish_call(
        content,
        parse_analysis,
        latency,
        usage,
        settings.model_1_model,
        settings.model_1_input_cost,
        settings.model_1_output_cost,
        "DeepSeek",
    )


def validate_with_model2(text, category, extraction, analysis, settings):
    user = (
        f"Route: {category}\n\n"
        "Extracted JSON:\n"
        f"{dump_json(extraction)}\n\n"
        "Analysis JSON:\n"
        f"{dump_json(analysis)}\n\n"
        "Document:\n"
        f"{clip_text(text)}"
    )
    content, latency, usage = claude_messages(
        settings.model_2_api_key,
        settings.model_2_model,
        VALIDATE_SYSTEM,
        user,
        settings,
    )
    return finish_call(
        content,
        parse_validation,
        latency,
        usage,
        settings.model_2_model,
        settings.model_2_input_cost,
        settings.model_2_output_cost,
        "Claude",
    )
