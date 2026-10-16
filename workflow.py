"""LangGraph workflow for the classroom finance demo.

Graph: classify -> one extraction path -> analyze -> validate -> complete.
Classification chooses the extraction path. A failed local or API call stops
later model calls and records the error. JEV failure uses the rule fallback.
"""

import contextvars
import io
import operator
import re
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

from models import (
    MAX_PROMPT_CHARS,
    ConfigError,
    ModelError,
    analyze_with_model1,
    blank_metrics,
    classify_with_jev,
    extract_with_ollama,
    failed_metric,
    load_settings,
    validate_with_model2,
)

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
SUPPORTED_SUFFIXES = {".txt", ".csv", ".pdf"}
MAX_BYTES = 2_000_000
CATEGORY_LABELS = {
    "invoice": "Invoice",
    "expense": "Expense",
    "financial_report": "Financial report",
    "other": "Other",
}
ROUTE_TO_NODE = {
    "invoice": "extract_invoice",
    "expense": "extract_expense",
    "financial_report": "extract_financial_report",
    "other": "extract_other",
}
NODE_LABELS = {
    "classify": "Classification",
    "extract_invoice": "Document extraction (invoice)",
    "extract_expense": "Document extraction (expense)",
    "extract_financial_report": "Document extraction (financial report)",
    "extract_other": "Document extraction (other)",
    "analyze": "Financial analysis",
    "validate": "Validation",
    "complete": "Complete",
}
# First match wins a tie. Phrases are matched once, on word boundaries.
KEYWORDS = {
    "invoice": (
        ("supplier invoice", 3),
        ("invoice number", 3),
        ("amount due", 2),
        ("subtotal", 2),
        ("invoice", 1),
    ),
    "expense": (
        ("expense record", 3),
        ("reimbursement", 3),
        ("employee", 2),
        ("expense", 2),
    ),
    "financial_report": (
        ("financial report", 3),
        ("net income", 3),
        ("gross profit", 2),
        ("operating income", 2),
        ("revenue", 2),
    ),
}
PRIORITY = ("invoice", "expense", "financial_report")

_progress = contextvars.ContextVar("finance_demo_progress", default=None)


class WorkflowState(TypedDict):
    document_name: str
    document_text: str
    route: str
    classification: dict
    extraction: dict
    analysis: dict
    validation: dict
    halt: bool
    step_status: dict
    metrics: dict
    logs: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    execution_order: Annotated[list, operator.add]


def set_progress_hook(fn):
    return _progress.set(fn)


def reset_progress_hook(token):
    _progress.reset(token)


def notify(step, status):
    fn = _progress.get()
    if fn is not None:
        fn(step, status)


def fresh_step_status():
    return {
        "classification": "pending",
        "extraction": "pending",
        "analysis": "pending",
        "validation": "pending",
        "complete": "pending",
    }


def error_item(step, message, fatal):
    return {"step": step, "message": message, "fatal": fatal}


def format_execution_order(order):
    if not order:
        return "not started"
    return " -> ".join(NODE_LABELS.get(item, item) for item in order)


def metric_line(label, metric):
    latency = metric.get("latency_s")
    if isinstance(latency, (int, float)):
        latency_text = f"{latency:.2f}s"
    else:
        latency_text = "unavailable"
    def token(name):
        value = metric.get(name)
        return "unavailable" if value is None else str(value)
    cost = metric.get("estimated_cost_usd")
    if label == "Ollama":
        cost_text = "not applicable"
    elif cost is None:
        cost_text = "unavailable"
    else:
        cost_text = f"{cost:.8f}".rstrip("0").rstrip(".") + " USD"
    model = metric.get("model") or "not set"
    return (
        f"{label} ({model}) finished in {latency_text}. "
        f"Tokens: prompt {token('prompt_tokens')}, completion {token('completion_tokens')}, "
        f"total {token('total_tokens')}. Estimated cost: {cost_text}."
    )


def contains_phrase(text, phrase):
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


def rule_classify(text):
    folded = text.casefold()
    scores = {}
    matched = {}
    for category, phrases in KEYWORDS.items():
        score = 0
        hits = []
        for phrase, weight in phrases:
            if contains_phrase(folded, phrase):
                score += weight
                hits.append(phrase)
        scores[category] = score
        matched[category] = hits
    best = max(scores, key=lambda category: (scores[category], -PRIORITY.index(category)))
    if scores[best] == 0:
        return {
            "category": "other",
            "confidence": None,
            "reason": "Rule-based fallback found no invoice, expense, or financial report keywords.",
            "keyword_scores": scores,
        }
    reason = (
        "Rule-based fallback. "
        f"Scores: invoice={scores['invoice']}, expense={scores['expense']}, "
        f"financial_report={scores['financial_report']}. "
        f"Matched: {', '.join(matched[best])}."
    )
    tied = [category for category, score in scores.items() if score == scores[best]]
    if len(tied) > 1:
        reason += " Tie broken by category priority: invoice, expense, financial_report."
    return {
        "category": best,
        "confidence": None,
        "reason": reason,
        "keyword_scores": scores,
    }


def attach_source(raw, source):
    category = raw["category"]
    result = {
        "category": category,
        "label": CATEGORY_LABELS[category],
        "confidence": raw.get("confidence"),
        "reason": raw.get("reason"),
        "source": source,
    }
    if "keyword_scores" in raw:
        result["keyword_scores"] = raw["keyword_scores"]
    return result


def list_samples():
    if not DOCUMENTS_DIR.is_dir():
        return []
    names = [
        path.name
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    preferred = {
        "sample_invoice.txt": 0,
        "sample_expense.csv": 1,
        "sample_report.txt": 2,
    }
    return sorted(names, key=lambda name: (preferred.get(name, 100), name))


def sample_path(filename):
    root = DOCUMENTS_DIR.resolve()
    candidate = (root / Path(filename).name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise ValueError(f"Sample document not found: {filename}")
    if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported sample file type: {filename}")
    return candidate


def load_sample(filename):
    path = sample_path(filename)
    return path.name, load_document_bytes(path.name, path.read_bytes())


def load_document_bytes(name, data):
    if len(data) > MAX_BYTES:
        raise ValueError(f"{name} is larger than 2 MB. This demo only accepts small documents.")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        shown = suffix or "none"
        raise ValueError(f"Unsupported file type '{shown}' for {name}. Use a PDF, CSV, or TXT file.")
    if suffix == ".pdf":
        text = pdf_text(name, data)
    else:
        text = data.decode("utf-8-sig", errors="replace")
    if not text.strip():
        raise ValueError(f"No text could be extracted from {name}.")
    return text


def pdf_text(name, data):
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError(f"{name} is encrypted. Export an unencrypted PDF and try again.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read {name} as a PDF: {exc}") from exc
    return "\n".join(pages)


def build_initial_state(document_name, document_text):
    if not document_text or not str(document_text).strip():
        raise ValueError("The document has no extracted text.")
    return {
        "document_name": document_name,
        "document_text": document_text,
        "route": "",
        "classification": {},
        "extraction": {},
        "analysis": {},
        "validation": {},
        "halt": False,
        "step_status": fresh_step_status(),
        "metrics": blank_metrics(),
        "logs": [f"Loaded {document_name} ({len(document_text)} characters)."],
        "errors": [],
        "execution_order": [],
    }


def fail_step(state, step_key, order_name, message, metric_key=None, metric=None):
    status = dict(state["step_status"])
    status[step_key] = "failed"
    metrics = dict(state["metrics"])
    if metric_key is not None and metric is not None:
        metrics[metric_key] = metric
    return {
        "halt": True,
        "step_status": status,
        "metrics": metrics,
        "logs": [message],
        "errors": [error_item(step_key, message, True)],
        "execution_order": [order_name],
    }


def skip_step(state, step_key, order_name, label):
    status = dict(state["step_status"])
    status[step_key] = "failed"
    reason = "an earlier step failed"
    for item in state.get("errors") or []:
        if item.get("fatal"):
            reason = item["message"]
            break
    return {
        "step_status": status,
        "logs": [f"{label} was not called because an earlier step failed: {reason}"],
        "execution_order": [order_name],
    }


def with_cost_error(result, step, cost_error):
    if cost_error:
        result["errors"] = [error_item(step, cost_error, False)]
    return result


def classify_node(state):
    notify("classification", "running")
    text = state["document_text"]
    logs = []
    if len(text) > MAX_PROMPT_CHARS:
        logs.append(
            f"Model prompts include only the first {MAX_PROMPT_CHARS} characters of the document."
        )
    try:
        settings = load_settings()
    except ConfigError as exc:
        result = fail_step(state, "classification", "classify", str(exc), "jev", failed_metric(None, None))
        result["logs"] = logs + result["logs"]
        return result
    try:
        parsed, metric, cost_error = classify_with_jev(text, settings)
    except ModelError as exc:
        classification = attach_source(rule_classify(text), "rule_fallback")
        message = f"JEV classification failed: {exc} Using the rule-based fallback classifier."
        status = dict(state["step_status"])
        status["classification"] = "completed"
        metrics = dict(state["metrics"])
        metrics["jev"] = failed_metric(settings.jev_model or None, exc.latency_s)
        return {
            "classification": classification,
            "route": classification["category"],
            "step_status": status,
            "metrics": metrics,
            "logs": logs
            + [
                message,
                (
                    "Rule-based fallback classified the document as "
                    f"{classification['category']}. Processing path: {classification['category']}."
                ),
                classification.get("reason") or "Rule-based fallback produced no reason.",
            ],
            "errors": [error_item("classification", message, False)],
            "execution_order": ["classify"],
        }
    classification = attach_source(parsed, "jev")
    status = dict(state["step_status"])
    status["classification"] = "completed"
    metrics = dict(state["metrics"])
    metrics["jev"] = metric
    result = {
        "classification": classification,
        "route": classification["category"],
        "step_status": status,
        "metrics": metrics,
        "logs": logs
        + [
            (
                f"JEV classified the document as {classification['category']}. "
                f"Processing path: {classification['category']}."
            ),
            metric_line("JEV", metric),
        ],
        "execution_order": ["classify"],
    }
    return with_cost_error(result, "classification", cost_error)


def make_extract(category, order_name):
    def node(state):
        if state.get("halt"):
            return skip_step(state, "extraction", order_name, "Document extraction")
        notify("extraction", "running")
        try:
            settings = load_settings()
        except ConfigError as exc:
            return fail_step(
                state, "extraction", order_name, str(exc), "ollama", failed_metric(None, None)
            )
        try:
            extraction, metric, cost_error = extract_with_ollama(
                state["document_text"], category, settings
            )
        except ModelError as exc:
            return fail_step(
                state,
                "extraction",
                order_name,
                str(exc),
                "ollama",
                failed_metric(settings.ollama_model, exc.latency_s),
            )
        status = dict(state["step_status"])
        status["extraction"] = "completed"
        metrics = dict(state["metrics"])
        metrics["ollama"] = metric
        result = {
            "extraction": extraction,
            "step_status": status,
            "metrics": metrics,
            "logs": [
                f"Document extraction path: {order_name}.",
                metric_line("Ollama", metric),
            ],
            "execution_order": [order_name],
        }
        return with_cost_error(result, "extraction", cost_error)

    return node


extract_invoice = make_extract("invoice", "extract_invoice")
extract_expense = make_extract("expense", "extract_expense")
extract_financial_report = make_extract("financial_report", "extract_financial_report")
extract_other = make_extract("other", "extract_other")


def analyze_node(state):
    if state.get("halt"):
        return skip_step(state, "analysis", "analyze", "Financial analysis")
    notify("analysis", "running")
    try:
        settings = load_settings()
    except ConfigError as exc:
        return fail_step(state, "analysis", "analyze", str(exc), "model_1", failed_metric(None, None))
    try:
        analysis, metric, cost_error = analyze_with_model1(
            state["document_text"],
            state.get("route") or "",
            state.get("extraction") or {},
            settings,
        )
    except ModelError as exc:
        return fail_step(
            state,
            "analysis",
            "analyze",
            str(exc),
            "model_1",
            failed_metric(settings.model_1_model, exc.latency_s),
        )
    status = dict(state["step_status"])
    status["analysis"] = "completed"
    metrics = dict(state["metrics"])
    metrics["model_1"] = metric
    result = {
        "analysis": analysis,
        "step_status": status,
        "metrics": metrics,
        "logs": [
            "Financial analysis was requested from DeepSeek.",
            metric_line("DeepSeek", metric),
        ],
        "execution_order": ["analyze"],
    }
    return with_cost_error(result, "analysis", cost_error)


def validate_node(state):
    if state.get("halt"):
        return skip_step(state, "validation", "validate", "Validation")
    notify("validation", "running")
    try:
        settings = load_settings()
    except ConfigError as exc:
        return fail_step(
            state, "validation", "validate", str(exc), "model_2", failed_metric(None, None)
        )
    try:
        validation, metric, cost_error = validate_with_model2(
            state["document_text"],
            state.get("route") or "",
            state.get("extraction") or {},
            state.get("analysis") or {},
            settings,
        )
    except ModelError as exc:
        return fail_step(
            state,
            "validation",
            "validate",
            str(exc),
            "model_2",
            failed_metric(settings.model_2_model, exc.latency_s),
        )
    status = dict(state["step_status"])
    status["validation"] = "completed"
    metrics = dict(state["metrics"])
    metrics["model_2"] = metric
    result = {
        "validation": validation,
        "step_status": status,
        "metrics": metrics,
        "logs": [
            "Validation was requested from Claude. This check is advisory, not proof of financial correctness.",
            f"Claude status: {validation['status']}.",
            metric_line("Claude", metric),
        ],
        "execution_order": ["validate"],
    }
    return with_cost_error(result, "validation", cost_error)


def finish_node(state):
    status = dict(state["step_status"])
    logs = []
    pending = [
        name
        for name in ("extraction", "analysis", "validation")
        if status.get(name) == "pending"
    ]
    if pending:
        for name in pending:
            status[name] = "failed"
        logs.append("These steps were not run: " + ", ".join(pending) + ".")
    failed = bool(state.get("halt")) or any(
        item.get("fatal") for item in state.get("errors") or []
    )
    failed = failed or any(
        status.get(name) == "failed"
        for name in ("classification", "extraction", "analysis", "validation")
    )
    status["complete"] = "failed" if failed else "completed"
    logs.append("Workflow finished with errors." if failed else "Workflow finished.")
    return {
        "step_status": status,
        "logs": logs,
        "execution_order": ["complete"],
    }


def select_route(state):
    if state.get("halt"):
        return "halted"
    route = state.get("route") or ""
    if route not in ROUTE_TO_NODE:
        return "other"
    return route


def build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("classify", classify_node)
    builder.add_node("extract_invoice", extract_invoice)
    builder.add_node("extract_expense", extract_expense)
    builder.add_node("extract_financial_report", extract_financial_report)
    builder.add_node("extract_other", extract_other)
    builder.add_node("analyze", analyze_node)
    builder.add_node("validate", validate_node)
    builder.add_node("complete", finish_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        select_route,
        {
            "invoice": "extract_invoice",
            "expense": "extract_expense",
            "financial_report": "extract_financial_report",
            "other": "extract_other",
            "halted": "complete",
        },
    )
    for name in (
        "extract_invoice",
        "extract_expense",
        "extract_financial_report",
        "extract_other",
    ):
        builder.add_edge(name, "analyze")
    builder.add_edge("analyze", "validate")
    builder.add_edge("validate", "complete")
    builder.add_edge("complete", END)
    return builder.compile()


graph = build_graph()
