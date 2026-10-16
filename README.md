# Finance Workflow Demo

Classroom demo of a small financial-document workflow. It is not an accounting system, and it does not post anything to a ledger.

The page runs one LangGraph pipeline:

1. Load a PDF, CSV, or TXT file.
2. JEV classifies it as an invoice, an expense, a financial report, or other.
3. The selected category routes the document to a matching extraction path.
4. A local Ollama model extracts financial fields as JSON.
5. DeepSeek analyzes that JSON.
6. Claude checks the extraction and the analysis.
7. The page shows each step, the results, and any usage figures the providers actually returned.

If JEV fails, a visible rule-based fallback chooses the route. If Ollama or either API model fails, the page shows the error and does not substitute another model or a canned success.

Validation is an advisory model check. It is not proof that the figures are financially correct.

## Models

| Role | Provider | What it does |
| --- | --- | --- |
| Classification | JEV | Chooses the processing path |
| Extraction | Ollama on CPU | Returns structured financial fields |
| Analysis | DeepSeek `deepseek-flash` | Insights, anomalies, missing information, risks |
| Validation | Claude `claude-sonnet-5-5` | Arithmetic, consistency, unsupported claims, missing fields |

Document text is sent to JEV, DeepSeek, and Claude. The local model also receives the document text. Use the fictional samples unless you intend to share other documents with those providers.

## Requirements

- Windows
- Python 3.11 or newer (these commands were checked with Python 3.11.9)
- [Ollama for Windows](https://ollama.com/download/windows)
- API credentials for JEV, DeepSeek, and Claude

The API values are placeholders until you add them. The app starts without them and reports each missing value when that step runs.

## 1. Create the environment and install dependencies

Open PowerShell and run:

```powershell
cd C:\Users\rezas\Desktop\accounting\finance-agent-demo
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`Set-ExecutionPolicy` applies only to the current PowerShell window. It lets the virtual environment activate.

## 2. Configure credentials

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in the JEV, DeepSeek, and Claude keys. Leave a value blank until you have it. Do not commit `.env`.

None of the three remote models uses a base URL from `.env`. JEV calls `https://api.typesafe.ai/v1/systemone` as `jev-latest` and asks one choice question: invoice, expense, financial report, or other. DeepSeek calls `https://api.deepseek.com/chat/completions` as `deepseek-flash`, with thinking disabled. Claude calls `https://api.anthropic.com/v1/messages` as `claude-sonnet-5-5`. Change a `*_MODEL` value only to pick a different model from that same provider.

USD prices per 1,000,000 tokens (`JEV_INPUT_USD_PER_MILLION`, `JEV_OUTPUT_USD_PER_MILLION`, and the matching `MODEL_1_*` and `MODEL_2_*` pairs) enable an estimated cost. The sample values are the published list prices: Jev $0.042 input and $0 output, DeepSeek flash off-peak cache-miss $0.15 input and $0.60 output, Claude Sonnet 5.5 $2 input and $10 output. Leave a pair blank and that row shows the cost as unavailable. The app does not guess token counts from text length. Ollama has no API cost in the comparison table.

`REQUEST_TIMEOUT_SECONDS` defaults to 180 because a small CPU model can be slow.

Saving `.env` is enough. The next Run Workflow reloads it. You do not need to restart Streamlit.

## 3. Install the local model

Install Ollama from <https://ollama.com/download/windows>. After the installer finishes, Ollama runs in the background. Open a new PowerShell window if `ollama` is not recognized, then run:

```powershell
ollama pull qwen2.5:0.5b
ollama list
```

The pull name must match `OLLAMA_MODEL` in `.env`. To use a different small model, change `OLLAMA_MODEL` and pull that name. The workflow will not switch models on its own.

If you opened a new terminal, activate the project environment again before the next step:

```powershell
cd C:\Users\rezas\Desktop\accounting\finance-agent-demo
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Start the app

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Use the page

The page has three sections.

**A. Input.** Choose a sample file or upload a PDF, CSV, or TXT file. The selected name is shown above **Run Workflow**.

**B. Execution.** The status list follows classification, document extraction, financial analysis, validation, and complete. Logs show the execution order, the fallback when JEV fails, and errors. Fatal errors stay on the page until you run another document.

**C. Results and Comparison.** Classification, extracted fields, analysis, validation status, total time, and a model table. Latency, token counts, and estimated cost appear only when that call produced them.

Run a second document from the same page. The app does not need a restart.

Prompts sent to the models include at most the first 8,000 characters. The full extracted text remains in the source expander.

Uploaded files must be PDF, CSV, or TXT and at most 2 MB.

## Sample documents

All three files are fictional and use EUR.

| File | What it is | Figures |
| --- | --- | --- |
| `documents/sample_invoice.txt` | Harbor Supply Co. invoice INV-2026-0315, dated 2026-03-15 | 430.00 + 90.30 tax = 520.30 |
| `documents/sample_expense.csv` | Metro Transit rail fare for Alex Morgan, dated 2026-03-18 | 48.00 + 10.08 tax = 58.08 |
| `documents/sample_report.txt` | Northwind Teaching Co. monthly report for March 2026 | Supplies 520.30 and travel 58.08 are included in operating expenses 2,480.00. Net income is 3,918.40. |

## Errors you should see before credentials are filled in

- Missing JEV settings: the log says JEV failed, then the rule-based fallback classifies the file.
- Ollama is not running, or `qwen2.5:0.5b` is not pulled: extraction fails and names the `ollama pull` command. Analysis and validation are not called.
- A remote model returns an error or unusable JSON: that step fails with the provider status or a JSON message. API keys are removed from the message.

Click **Run Workflow** to retry.

## Checks

From the project folder, with the virtual environment active:

```powershell
python -m compileall -q .
python -m unittest test_core.py
```

## Project layout

```text
finance-agent-demo/
├── app.py
├── workflow.py
├── models.py
├── test_core.py
├── requirements.txt
├── .env.example
├── documents/
│   ├── sample_invoice.txt
│   ├── sample_expense.csv
│   └── sample_report.txt
└── README.md
```

`workflow.py` is the LangGraph pipeline. `models.py` is the only place that calls JEV, Ollama, and the two API models. `app.py` is the Streamlit page.
