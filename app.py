"""One-page Streamlit dashboard for the classroom finance workflow."""

import copy
import time

import streamlit as st

from models import ConfigError, blank_metric, load_settings, public_config_line
from workflow import (
    build_initial_state,
    error_item,
    format_execution_order,
    fresh_step_status,
    graph,
    list_samples,
    load_document_bytes,
    load_sample,
    reset_progress_hook,
    set_progress_hook,
)

STEP_ROWS = (
    ("classification", "Classification"),
    ("extraction", "Document extraction"),
    ("analysis", "Financial analysis"),
    ("validation", "Validation"),
    ("complete", "Complete"),
)
STEP_RUNNING = {
    "classification": "Classification is running.",
    "extraction": "Document extraction is running.",
    "analysis": "Financial analysis is running.",
    "validation": "Validation is running.",
    "complete": "Complete is running.",
}
COMPARISON_ROWS = (
    ("jev", "Classification", "JEV"),
    ("ollama", "Document extraction", "Ollama"),
    ("model_1", "Financial analysis", "DeepSeek"),
    ("model_2", "Validation", "Claude"),
)


def idle_state():
    return {
        "document_name": "",
        "document_text": "",
        "route": "",
        "classification": {},
        "extraction": {},
        "analysis": {},
        "validation": {},
        "halt": False,
        "step_status": fresh_step_status(),
        "metrics": {},
        "logs": [],
        "errors": [],
        "execution_order": [],
    }


def failed_start(name, message):
    state = idle_state()
    state["document_name"] = name or ""
    state["logs"] = [message, "The workflow did not start."]
    state["errors"] = [error_item("input", message, True)]
    state["step_status"]["complete"] = "failed"
    return {"state": state, "total_seconds": None, "document_name": name or ""}


def selection_name(selection):
    if selection is None:
        return None
    kind, payload = selection
    if kind == "sample":
        return payload
    return payload.name


def resolve_document(selection):
    kind, payload = selection
    if kind == "sample":
        return load_sample(payload)
    return payload.name, load_document_bytes(payload.name, payload.getvalue())


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def latency_cell(metric):
    if metric.get("status") == "not_called":
        return "not called"
    latency = metric.get("latency_s")
    if not isinstance(latency, (int, float)):
        return "unavailable"
    return f"{latency:.2f}"


def token_cell(metric, key):
    if metric.get("status") == "not_called":
        return "not called"
    value = metric.get(key)
    if value is None:
        return "unavailable"
    return str(value)


def cost_cell(metric, provider):
    if metric.get("status") == "not_called":
        return "not called"
    if provider == "Ollama":
        return "not applicable"
    cost = metric.get("estimated_cost_usd")
    if cost is None:
        return "unavailable"
    return f"{cost:.8f}".rstrip("0").rstrip(".")


def model_cell(metric):
    if metric.get("status") == "not_called":
        return "not called"
    return metric.get("model") or "not set"


def show_config():
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(str(exc))
        return
    st.caption(public_config_line(settings))


def show_execution(target, state, running_note=None):
    logs = list(state.get("logs") or [])
    if running_note:
        logs.append(running_note)
    status = state.get("step_status") or {}
    warnings = [item for item in state.get("errors") or [] if not item.get("fatal")]
    fatals = [item for item in state.get("errors") or [] if item.get("fatal")]
    with target.container():
        for index, (key, label) in enumerate(STEP_ROWS, start=1):
            st.write(f"{index}. {label} — {status.get(key, 'pending')}")
        st.write("Execution order: " + format_execution_order(state.get("execution_order") or []))
        st.markdown("**Logs**")
        if logs:
            for line in logs:
                st.write(f"- {line}")
        else:
            st.write("No logs yet.")
        for item in warnings:
            st.warning(item["message"])
        for item in fatals:
            st.error(item["message"])
        if fatals:
            st.write("Click Run Workflow to try again. You do not need to restart the app.")


def show_json_section(title, status_name, payload, state):
    st.subheader(title)
    if state.get("step_status", {}).get(status_name) == "completed" and payload:
        st.json(payload)
    else:
        st.write("Not available.")


def show_results(target, state, total_seconds, current_name):
    with target.container():
        if state is None:
            st.write("Run the workflow to see results here.")
            return
        document_name = state.get("document_name") or ""
        if document_name and current_name and document_name != current_name:
            st.caption(
                f"Showing results for {document_name}. The current selection is {current_name}. "
                "Click Run Workflow to process the current selection."
            )
        st.write(f"Document: {document_name or 'not available'}")
        if isinstance(total_seconds, (int, float)):
            st.write(f"Total execution time: {total_seconds:.2f} seconds")
        else:
            st.write("Total execution time: not available")

        classification = state.get("classification") or {}
        st.subheader("Document classification")
        if state.get("step_status", {}).get("classification") == "completed" and classification:
            source = classification.get("source")
            if source == "jev":
                source_label = "JEV"
            elif source == "rule_fallback":
                source_label = "rule-based fallback"
            else:
                source_label = source or "not provided"
            confidence = classification.get("confidence")
            confidence_text = "not provided" if confidence is None else f"{confidence:.2f}"
            reason = classification.get("reason")
            st.write(f"Category: {classification.get('label') or classification.get('category')}")
            st.write(f"Source: {source_label}")
            st.write(f"Processing path: {state.get('route') or 'not set'}")
            st.write(f"Confidence: {confidence_text}")
            st.write(f"Reason: {reason if reason else 'not provided'}")
            st.json(classification)
        else:
            st.write("Not available.")

        show_json_section(
            "Extracted financial fields",
            "extraction",
            state.get("extraction") or {},
            state,
        )
        show_json_section(
            "Financial analysis",
            "analysis",
            state.get("analysis") or {},
            state,
        )

        st.subheader("Validation status")
        st.caption("This validation is an advisory model check, not proof of financial correctness.")
        validation = state.get("validation") or {}
        if state.get("step_status", {}).get("validation") == "completed" and validation:
            st.write(f"Status: {validation.get('status')}")
            st.write(validation.get("explanation") or "")
            st.json(validation)
        else:
            st.write("Not available.")

        st.subheader("Model comparison")
        metrics = state.get("metrics") or {}
        header = (
            "| Role | Provider | Model | Status | Latency (s) | Prompt tokens | "
            "Completion tokens | Total tokens | Estimated cost (USD) |"
        )
        divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        lines = [header, divider]
        for key, role, provider in COMPARISON_ROWS:
            metric = metrics.get(key) or blank_metric()
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_cell(role),
                        markdown_cell(provider),
                        markdown_cell(model_cell(metric)),
                        markdown_cell(str(metric.get("status") or "not_called").replace("_", " ")),
                        markdown_cell(latency_cell(metric)),
                        markdown_cell(token_cell(metric, "prompt_tokens")),
                        markdown_cell(token_cell(metric, "completion_tokens")),
                        markdown_cell(token_cell(metric, "total_tokens")),
                        markdown_cell(cost_cell(metric, provider)),
                    ]
                )
                + " |"
            )
        st.markdown("\n".join(lines))
        st.caption(
            "Token counts appear only when the provider returns them. "
            "Estimated cost appears only when those counts and both USD prices in .env are available. "
            "Costs are not guessed from document length. "
            "Ollama runs locally, so this table does not give it an API cost. "
            "Ollama token counts are prompt_eval_count and eval_count when Ollama returns them."
        )
        if state.get("document_text"):
            with st.expander("Source text"):
                st.text(state["document_text"])


def execute(selection, execution_box, results_box):
    current_name = selection_name(selection)
    try:
        name, text = resolve_document(selection)
    except ValueError as exc:
        return failed_start(current_name, str(exc))

    state = build_initial_state(name, text)
    holder = {"state": state}
    started = time.perf_counter()

    def on_progress(step, status):
        view = copy.deepcopy(holder["state"])
        view["step_status"] = dict(view["step_status"])
        view["step_status"][step] = status
        note = STEP_RUNNING.get(step) if status == "running" else None
        elapsed = time.perf_counter() - started
        show_execution(execution_box, view, note)
        show_results(results_box, view, elapsed, name)

    token = set_progress_hook(on_progress)
    try:
        on_progress("classification", "running")
        final = state
        for snapshot in graph.stream(state, stream_mode="values"):
            final = snapshot
            holder["state"] = snapshot
            elapsed = time.perf_counter() - started
            show_execution(execution_box, snapshot)
            show_results(results_box, snapshot, elapsed, name)
    except Exception as exc:
        final = copy.deepcopy(holder["state"])
        final["halt"] = True
        final["errors"] = list(final.get("errors") or []) + [
            error_item("workflow", f"The workflow stopped unexpectedly: {exc}", True)
        ]
        final["logs"] = list(final.get("logs") or []) + [
            f"The workflow stopped unexpectedly: {exc}"
        ]
        status = dict(final["step_status"])
        status["complete"] = "failed"
        final["step_status"] = status
    finally:
        reset_progress_hook(token)
    return {
        "state": final,
        "total_seconds": time.perf_counter() - started,
        "document_name": name,
    }


def choose_document():
    samples = list_samples()
    source = st.radio("Document source", ["Sample file", "Upload"], horizontal=True)
    if source == "Sample file":
        if not samples:
            st.error("No PDF, CSV, or TXT files were found in the documents folder.")
            return None
        name = st.selectbox("Sample document", samples)
        st.write(f"Selected document: {name}")
        st.caption("Sample files are fictional and are not real accounting records.")
        return ("sample", name)
    upload = st.file_uploader("Upload a PDF, CSV, or TXT file", type=["pdf", "csv", "txt"])
    if upload is None:
        st.write("Selected document: none")
        return None
    st.write(f"Selected document: {upload.name}")
    return ("upload", upload)


def main():
    st.set_page_config(page_title="Finance Workflow Demo", layout="wide")
    st.title("Finance Workflow Demo")
    st.write(
        "This classroom demo classifies a document, extracts fields with a local Ollama model, "
        "analyzes them with DeepSeek, and checks that work with Claude. "
        "Sample files are fictional. Document text is sent to the configured remote APIs. "
        "Validation is an advisory model check, not proof of financial correctness."
    )

    st.header("A. Input")
    selection = choose_document()
    show_config()
    run = st.button("Run Workflow", type="primary")

    st.header("B. Execution")
    execution_box = st.empty()
    st.header("C. Results and Comparison")
    results_box = st.empty()

    if "result" not in st.session_state:
        st.session_state.result = None

    if run:
        if selection is None:
            st.session_state.result = failed_start(
                "",
                "Select a sample document or upload a PDF, CSV, or TXT file.",
            )
        else:
            st.session_state.result = execute(selection, execution_box, results_box)

    result = st.session_state.result
    current_name = selection_name(selection)
    if result is None:
        show_execution(execution_box, idle_state())
        show_results(results_box, None, None, current_name)
    else:
        show_execution(execution_box, result["state"])
        show_results(results_box, result["state"], result["total_seconds"], current_name)


if __name__ == "__main__":
    main()
