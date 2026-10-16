"""Checks that do not call a live model or invent a successful workflow run."""

import csv
import io
import json
import re
import unittest
from unittest import mock

import requests

from models import (
    ANALYSIS_KEYS,
    VALIDATION_LIST_KEYS,
    ConfigError,
    ModelError,
    analyze_with_model1,
    build_metric,
    classify_with_jev,
    estimate_cost,
    extract_with_ollama,
    parse_analysis,
    parse_classification,
    parse_extraction,
    parse_validation,
    public_config_line,
    validate_with_model2,
    scrub,
    settings_from_mapping,
)
from workflow import (
    build_graph,
    build_initial_state,
    finish_node,
    load_document_bytes,
    load_sample,
    rule_classify,
    select_route,
)


def extraction_payload(**overrides):
    payload = {
        "document_type": "invoice",
        "date": "2026-03-15",
        "vendor": "Harbor Supply Co.",
        "amount": "520.30",
        "currency": "EUR",
        "tax": "90.30",
        "description": "Office supplies",
    }
    payload.update(overrides)
    return payload


def analysis_payload():
    return {key: ["note"] for key in ANALYSIS_KEYS}


def validation_payload():
    payload = {key: ["note"] for key in VALIDATION_LIST_KEYS}
    payload["status"] = "review"
    payload["explanation"] = "Checked the supplied figures."
    return payload


def json_response(content, usage=None, status=200):
    response = mock.Mock()
    response.status_code = status
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    response.json.return_value = body
    response.text = json.dumps(body)
    return response


class DemoTests(unittest.TestCase):
    def test_settings_defaults_and_timeout(self):
        settings = settings_from_mapping({})
        self.assertEqual(settings.ollama_model, "qwen2.5:0.5b")
        self.assertEqual(settings.model_1_model, "deepseek-flash")
        self.assertEqual(settings.model_2_model, "claude-sonnet-5-5")
        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(settings.jev_model, "jev-latest")
        self.assertEqual(settings.timeout_seconds, 180)
        self.assertEqual(settings.jev_api_key, "")
        with self.assertRaises(ConfigError):
            settings_from_mapping({"REQUEST_TIMEOUT_SECONDS": "fast"})

    def test_public_config_hides_keys(self):
        settings = settings_from_mapping(
            {
                "JEV_API_KEY": "sk-secret-value-123",
                "MODEL_1_API_KEY": "model-one-secret-key",
                "JEV_MODEL": "jev-demo",
            }
        )
        text = public_config_line(settings)
        self.assertNotIn("sk-secret-value-123", text)
        self.assertNotIn("model-one-secret-key", text)
        self.assertIn("JEV: key set", text)
        self.assertIn("API Model 2 (Claude): key not set", text)
        self.assertIn("deepseek-flash", text)
        self.assertIn("claude-sonnet-5-5", text)
        self.assertIn("jev-demo", text)

    def test_scrub_removes_bearer_and_keys(self):
        settings = settings_from_mapping(
            {
                "JEV_API_KEY": "sk-secret-value-123",
                "MODEL_1_API_KEY": "model-one-secret-key",
                "MODEL_2_API_KEY": "model-two-secret-key",
            }
        )
        text = scrub(
            "Bearer sk-secret-value-123 failed model-one-secret-key and model-two-secret-key",
            settings,
        )
        self.assertNotIn("sk-secret-value-123", text)
        self.assertNotIn("model-one-secret-key", text)
        self.assertNotIn("model-two-secret-key", text)
        self.assertIn("[redacted]", text)

    def test_cost_is_calculated_only_from_returned_usage_and_prices(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 2_000_000}
        cost, error = estimate_cost(usage, "0.5", "1.5", "API Model 1")
        self.assertIsNone(error)
        self.assertAlmostEqual(cost, 3.5)
        cost, error = estimate_cost(usage, "", "", "API Model 1")
        self.assertIsNone(cost)
        self.assertIsNone(error)
        cost, error = estimate_cost(None, "0.5", "1.5", "API Model 1")
        self.assertIsNone(cost)
        self.assertIsNone(error)
        cost, error = estimate_cost(usage, "0.5", "", "API Model 1")
        self.assertIsNone(cost)
        self.assertIn("both", error)
        cost, error = estimate_cost(usage, "nope", "1", "API Model 1")
        self.assertIsNone(cost)
        self.assertIn("numbers", error)

    def test_classification_parser_rejects_unknown_categories(self):
        parsed = parse_classification(
            {"category": "invoice", "confidence": "0.25", "reason": "It is an invoice."}
        )
        self.assertEqual(parsed["category"], "invoice")
        self.assertAlmostEqual(parsed["confidence"], 0.25)
        parsed = parse_classification({"category": "invoice", "confidence": 12, "reason": "x"})
        self.assertIsNone(parsed["confidence"])
        with self.assertRaises(ModelError):
            parse_classification({"category": "memo", "confidence": 0.2, "reason": "no"})

    def test_extraction_parser_requires_fields_and_reads_numbers(self):
        parsed = parse_extraction(extraction_payload())
        self.assertEqual(parsed["amount"], 520.30)
        self.assertEqual(parsed["tax"], 90.30)
        payload = extraction_payload()
        del payload["tax"]
        with self.assertRaises(ModelError) as caught:
            parse_extraction(payload)
        self.assertIn("tax", str(caught.exception))

    def test_analysis_and_validation_parsers(self):
        self.assertEqual(parse_analysis(analysis_payload())["insights"], ["note"])
        bad = analysis_payload()
        bad["risks"] = "one risk"
        with self.assertRaises(ModelError):
            parse_analysis(bad)
        self.assertEqual(parse_validation(validation_payload())["status"], "review")
        with self.assertRaises(ModelError):
            parse_validation({**validation_payload(), "status": "approved"})

    def test_missing_jev_key_does_not_call_the_network(self):
        settings = settings_from_mapping({})
        with mock.patch("models.requests.post") as post:
            with self.assertRaises(ConfigError) as caught:
                classify_with_jev("hello", settings)
            post.assert_not_called()
        self.assertIn("JEV_API_KEY", str(caught.exception))

    def test_jev_posts_systemone_choice(self):
        settings = settings_from_mapping({"JEV_API_KEY": "jev-secret-key-value"})
        body = {
            "model": "jev-1.13.0",
            "answers": {
                "category": {
                    "type": "choice",
                    "choice": "invoice",
                    "confidence": 0.91,
                    "probabilities": {
                        "invoice": 0.91,
                        "expense": 0.05,
                        "financial_report": 0.03,
                        "other": 0.01,
                    },
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = body
        response.text = json.dumps(body)
        with mock.patch("models.requests.post", return_value=response) as post:
            parsed, metric, cost_error = classify_with_jev("Invoice from Harbor", settings)
        self.assertEqual(post.call_args.args[0], "https://api.typesafe.ai/v1/systemone")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["model"], "jev-latest")
        self.assertEqual(sent["questions"]["category"]["type"], "choice")
        self.assertNotIn("messages", sent)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer jev-secret-key-value",
        )
        self.assertEqual(parsed["category"], "invoice")
        self.assertAlmostEqual(parsed["confidence"], 0.91)
        self.assertIn("invoice 0.91", parsed["reason"])
        self.assertEqual(metric["model"], "jev-1.13.0")
        self.assertEqual(metric["prompt_tokens"], 100)
        self.assertEqual(metric["completion_tokens"], 20)
        self.assertIsNone(cost_error)
        self.assertIsNone(metric["estimated_cost_usd"])

    def test_api_error_redacts_the_key(self):
        settings = settings_from_mapping(
            {
                "MODEL_1_API_KEY": "model-one-secret-key",
                "MODEL_1_INPUT_USD_PER_MILLION": "1",
                "MODEL_1_OUTPUT_USD_PER_MILLION": "2",
            }
        )
        response = mock.Mock()
        response.status_code = 401
        response.text = "bad key model-one-secret-key"
        response.json.side_effect = ValueError("not json")
        with mock.patch("models.requests.post", return_value=response):
            with self.assertRaises(ModelError) as caught:
                analyze_with_model1("text", "invoice", extraction_payload(), settings)
        self.assertNotIn("model-one-secret-key", str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    def test_model1_cost_uses_returned_usage_only(self):
        settings = settings_from_mapping(
            {
                "MODEL_1_API_KEY": "model-one-secret-key",
                "MODEL_1_INPUT_USD_PER_MILLION": "0.5",
                "MODEL_1_OUTPUT_USD_PER_MILLION": "1.5",
            }
        )
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        response = json_response(json.dumps(analysis_payload()), usage)
        with mock.patch("models.requests.post", return_value=response) as post:
            _parsed, metric, cost_error = analyze_with_model1(
                "text", "invoice", extraction_payload(), settings
            )
        self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/chat/completions")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["model"], "deepseek-flash")
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertIsNone(cost_error)
        self.assertEqual(metric["prompt_tokens"], 1000)
        self.assertEqual(metric["total_tokens"], 1500)
        self.assertAlmostEqual(metric["estimated_cost_usd"], (1000 * 0.5 + 500 * 1.5) / 1_000_000)

        response = json_response(json.dumps(analysis_payload()), usage=None)
        with mock.patch("models.requests.post", return_value=response):
            _parsed, metric, cost_error = analyze_with_model1(
                "text", "invoice", extraction_payload(), settings
            )
        self.assertIsNone(metric["estimated_cost_usd"])
        self.assertIsNone(metric["prompt_tokens"])
        self.assertIsNone(cost_error)

    def test_model2_uses_claude_messages(self):
        settings = settings_from_mapping({"MODEL_2_API_KEY": "model-two-secret-key"})
        body = {
            "content": [
                {"type": "thinking", "thinking": "check the figures"},
                {"type": "text", "text": json.dumps(validation_payload())},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = body
        response.text = json.dumps(body)
        with mock.patch("models.requests.post", return_value=response) as post:
            parsed, metric, cost_error = validate_with_model2(
                "text", "invoice", extraction_payload(), analysis_payload(), settings
            )
        self.assertEqual(post.call_args.args[0], "https://api.anthropic.com/v1/messages")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["model"], "claude-sonnet-5-5")
        self.assertNotIn("temperature", sent)
        self.assertEqual(sent["thinking"], {"type": "between_tools"})
        self.assertEqual(post.call_args.kwargs["headers"]["x-api-key"], "model-two-secret-key")
        self.assertEqual(post.call_args.kwargs["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(parsed["status"], "review")
        self.assertEqual(metric["prompt_tokens"], 10)
        self.assertEqual(metric["completion_tokens"], 4)
        self.assertEqual(metric["model"], "claude-sonnet-5-5")
        self.assertIsNone(cost_error)
        self.assertIsNone(metric["estimated_cost_usd"])

    def test_ollama_connection_and_missing_model_errors(self):
        settings = settings_from_mapping({})
        with mock.patch("models.requests.post", side_effect=requests.ConnectionError("down")):
            with self.assertRaises(ModelError) as caught:
                extract_with_ollama("invoice text", "invoice", settings)
        message = str(caught.exception)
        self.assertIn("Start the Ollama application", message)
        self.assertIn("ollama pull qwen2.5:0.5b", message)

        response = mock.Mock()
        response.status_code = 404
        response.json.return_value = {"error": "model 'qwen2.5:0.5b' not found"}
        response.text = "model not found"
        with mock.patch("models.requests.post", return_value=response):
            with self.assertRaises(ModelError) as caught:
                extract_with_ollama("invoice text", "invoice", settings)
        self.assertIn("ollama pull qwen2.5:0.5b", str(caught.exception))

    def test_ollama_maps_eval_counts_and_does_not_invent_cost(self):
        settings = settings_from_mapping({})
        body = {
            "message": {"content": json.dumps(extraction_payload(amount=520.30, tax=90.30))},
            "prompt_eval_count": 11,
            "eval_count": 7,
        }
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = body
        response.text = json.dumps(body)
        with mock.patch("models.requests.post", return_value=response) as post:
            parsed, metric, cost_error = extract_with_ollama("invoice text", "invoice", settings)
        self.assertEqual(parsed["amount"], 520.30)
        self.assertEqual(metric["prompt_tokens"], 11)
        self.assertEqual(metric["completion_tokens"], 7)
        self.assertEqual(metric["total_tokens"], 18)
        self.assertIsNone(metric["estimated_cost_usd"])
        self.assertIsNone(cost_error)
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen2.5:0.5b")

    def test_invalid_local_json_is_an_error(self):
        settings = settings_from_mapping({})
        body = {"message": {"content": "this is not json"}}
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = body
        response.text = json.dumps(body)
        with mock.patch("models.requests.post", return_value=response):
            with self.assertRaises(ModelError) as caught:
                extract_with_ollama("invoice text", "invoice", settings)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_sample_files_route_and_balance(self):
        expected = {
            "sample_invoice.txt": "invoice",
            "sample_expense.csv": "expense",
            "sample_report.txt": "financial_report",
        }
        for filename, category in expected.items():
            _name, text = load_sample(filename)
            self.assertIn("FICTIONAL", text.upper())
            classified = rule_classify(text)
            self.assertEqual(classified["category"], category, classified["reason"])

        _name, invoice = load_sample("sample_invoice.txt")
        subtotal = self.money(r"Subtotal:\s*([0-9]+\.[0-9]{2}) EUR", invoice)
        tax = self.money(r"Tax \(VAT 21%\):\s*([0-9]+\.[0-9]{2}) EUR", invoice)
        total = self.money(r"Total amount due:\s*([0-9]+\.[0-9]{2}) EUR", invoice)
        self.assertAlmostEqual(20 * 12.50 + 4 * 45.00, subtotal)
        self.assertAlmostEqual(subtotal + tax, total)
        self.assertAlmostEqual(total, 520.30)

        _name, expense_text = load_sample("sample_expense.csv")
        row = next(csv.DictReader(io.StringIO(expense_text)))
        expense_total = float(row["total_amount"])
        self.assertAlmostEqual(float(row["net_amount"]) + float(row["tax"]), expense_total)
        self.assertEqual(row["currency"], "EUR")

        _name, report = load_sample("sample_report.txt")

        def field(label):
            match = re.search(rf"^{re.escape(label)}: ([0-9]+\.[0-9]{{2}}) EUR\r?$", report, re.M)
            self.assertIsNotNone(match, label)
            return float(match.group(1))

        self.assertAlmostEqual(field("Revenue") - field("Cost of goods sold"), field("Gross profit"))
        self.assertAlmostEqual(
            field("Supplies") + field("Travel") + field("Other operating expenses"),
            field("Operating expenses"),
        )
        self.assertAlmostEqual(field("Gross profit") - field("Operating expenses"), field("Operating income"))
        self.assertAlmostEqual(round(field("Operating income") * 0.21, 2), field("Income tax"))
        self.assertAlmostEqual(field("Operating income") - field("Income tax"), field("Net income"))
        self.assertAlmostEqual(field("Supplies"), total)
        self.assertAlmostEqual(field("Travel"), expense_total)

    def money(self, pattern, text):
        match = re.search(pattern, text)
        self.assertIsNotNone(match, pattern)
        return float(match.group(1))

    def test_document_loader_rejects_bad_files(self):
        with self.assertRaises(ValueError) as caught:
            load_document_bytes("notes.docx", b"hello")
        self.assertIn("Unsupported", str(caught.exception))
        with self.assertRaises(ValueError):
            load_document_bytes("blank.txt", b"  \n")
        with self.assertRaises(ValueError) as caught:
            load_document_bytes("bad.pdf", b"%PDF-1.4\nnot a real pdf")
        self.assertIn("PDF", str(caught.exception))
        with self.assertRaises(ValueError):
            load_document_bytes("big.txt", b"a" * (2_000_001))
        with self.assertRaises(ValueError):
            load_sample("../secrets.txt")

    def test_route_selection(self):
        self.assertEqual(select_route({"halt": True, "route": "invoice"}), "halted")
        self.assertEqual(select_route({"halt": False, "route": "expense"}), "expense")
        self.assertEqual(select_route({"halt": False, "route": ""}), "other")
        self.assertEqual(select_route({"halt": False, "route": "payroll"}), "other")

    def test_graph_contains_the_route_nodes(self):
        raw = build_graph().get_graph().nodes
        if isinstance(raw, dict):
            node_ids = set(raw)
        else:
            node_ids = {getattr(node, "id", node) for node in raw}
        for name in (
            "classify",
            "extract_invoice",
            "extract_expense",
            "extract_financial_report",
            "extract_other",
            "analyze",
            "validate",
            "complete",
        ):
            self.assertIn(name, node_ids)

    def test_finish_marks_unrun_steps_failed(self):
        state = build_initial_state("a.txt", "hello")
        state["halt"] = True
        state["step_status"]["classification"] = "failed"
        state["errors"] = [{"step": "classification", "message": "bad config", "fatal": True}]
        out = finish_node(state)
        self.assertEqual(out["step_status"]["extraction"], "failed")
        self.assertEqual(out["step_status"]["analysis"], "failed")
        self.assertEqual(out["step_status"]["validation"], "failed")
        self.assertEqual(out["step_status"]["complete"], "failed")
        self.assertIn("not run", out["logs"][0])

    def _run_graph(self, text, classify, extract, analyze, validate):
        final = None
        with (
            mock.patch("workflow.load_settings", return_value=settings_from_mapping({})),
            mock.patch("workflow.classify_with_jev", classify),
            mock.patch("workflow.extract_with_ollama", extract),
            mock.patch("workflow.analyze_with_model1", analyze),
            mock.patch("workflow.validate_with_model2", validate),
            mock.patch("models.requests.post", side_effect=AssertionError("network was called")),
        ):
            for snapshot in build_graph().stream(
                build_initial_state("doc.txt", text),
                stream_mode="values",
            ):
                final = snapshot
        return final

    def test_graph_routes_each_category_and_keeps_metrics(self):
        categories = {
            "invoice": "extract_invoice",
            "expense": "extract_expense",
            "financial_report": "extract_financial_report",
            "other": "extract_other",
        }
        for category, node_name in categories.items():
            def classify(text, settings, category=category):
                return (
                    {"category": category, "confidence": 0.5, "reason": "stub"},
                    build_metric(
                        "completed",
                        "stub-jev",
                        0.1,
                        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                        None,
                    ),
                    None,
                )

            def extract(text, cat, settings, category=category):
                self.assertEqual(cat, category)
                return (
                    extraction_payload(document_type=category),
                    build_metric("completed", "stub-ollama", 0.2, None, None),
                    None,
                )

            def analyze(text, cat, extraction, settings, category=category):
                self.assertEqual(cat, category)
                return (
                    analysis_payload(),
                    build_metric(
                        "completed",
                        "stub-1",
                        0.3,
                        {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                        0.01,
                    ),
                    None,
                )

            def validate(text, cat, extraction, analysis, settings):
                return (
                    validation_payload(),
                    build_metric("completed", "stub-2", 0.4, None, None),
                    None,
                )

            final = self._run_graph("classroom text", classify, extract, analyze, validate)
            self.assertEqual(final["route"], category)
            self.assertEqual(
                final["execution_order"],
                ["classify", node_name, "analyze", "validate", "complete"],
            )
            self.assertEqual(final["step_status"]["complete"], "completed")
            self.assertEqual(final["validation"]["status"], "review")
            self.assertIsNone(final["metrics"]["ollama"]["estimated_cost_usd"])
            self.assertIsNone(final["metrics"]["jev"]["estimated_cost_usd"])
            self.assertEqual(final["metrics"]["model_1"]["estimated_cost_usd"], 0.01)
            self.assertIsNone(final["metrics"]["model_2"]["prompt_tokens"])

    def test_jev_failure_falls_back_and_ollama_failure_stops_later_models(self):
        _name, text = load_sample("sample_invoice.txt")

        def classify(text, settings):
            raise ConfigError("JEV API key is not set. Add JEV_API_KEY to .env and run the workflow again.")

        def extract(text, category, settings):
            raise ModelError(
                "Could not connect to Ollama at http://127.0.0.1:11434/api/chat.",
                latency_s=0.05,
            )

        def analyze(*args, **kwargs):
            raise AssertionError("API Model 1 should not be called")

        def validate(*args, **kwargs):
            raise AssertionError("API Model 2 should not be called")

        final = self._run_graph(text, classify, extract, analyze, validate)
        self.assertEqual(final["route"], "invoice")
        self.assertEqual(final["classification"]["source"], "rule_fallback")
        self.assertFalse(final["errors"][0]["fatal"])
        self.assertTrue(final["halt"])
        self.assertEqual(final["metrics"]["ollama"]["status"], "failed")
        self.assertEqual(final["metrics"]["model_1"]["status"], "not_called")
        self.assertEqual(final["metrics"]["model_2"]["status"], "not_called")
        self.assertEqual(final["step_status"]["analysis"], "failed")
        self.assertEqual(final["step_status"]["validation"], "failed")
        self.assertEqual(final["step_status"]["complete"], "failed")
        joined = "\n".join(final["logs"])
        self.assertIn("rule-based fallback", joined.lower())
        self.assertIn("was not called", joined.lower())
        self.assertIn("extract_invoice", final["execution_order"])


if __name__ == "__main__":
    unittest.main()
