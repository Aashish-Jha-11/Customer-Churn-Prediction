# Customer Churn Prediction & Agentic Retention Assistant

A web application that predicts customer churn with a Random Forest model and then uses an **agentic LLM workflow** (LangGraph + Groq Llama 3.3 70B + FAISS RAG) to generate structured, source-backed retention recommendations. Built with Streamlit.

See [EXPLANATION.md](EXPLANATION.md) for a component-by-component walkthrough.

## Features

- **Single Prediction** — enter one customer's details, get churn probability + rule-based reasons.
- **Batch Prediction** — upload a CSV, get predictions for the whole file, download as CSV.
- **Retention Report (NEW)** — agentic workflow produces a structured retention plan with sources; export as PDF.
- **Analytics Dashboard (NEW)** — portfolio-level Plotly charts, risk tiers, value-vs-risk segmentation.

## Installation

```bash
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

## Configure the LLM (free)

1. Sign up at [console.groq.com](https://console.groq.com) and create an API key (free tier).
2. Create `.streamlit/secrets.toml` locally (it's gitignored) with:
   ```toml
   GROQ_API_KEY = "gsk_your_key"
   ```
3. Without a key the Retention Report mode falls back to a rule-based report automatically.

## Running locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## CSV format (Batch + Dashboard)

```csv
Total Spend,Support Calls,Payment Delay,Contract Length
500.0,2,0,Monthly
1200.0,5,15,Annual
800.0,1,5,Quarterly
```

## Deploy free on Streamlit Community Cloud

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo + `app.py` on `main`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key"
   ```
4. Deploy. First build downloads MiniLM + FAISS + torch (~3 min).

`runtime.txt` pins Python 3.11 for reliable `faiss-cpu` wheels. `packages.txt` is empty (no apt packages needed).

## Running the smoke test

Offline, no API calls:

```bash
python -m tests.smoke_agent
```

Tests three customer profiles plus the LLM-unavailable fallback path.

## Model Information

- Model Type: Random Forest Classifier (scikit-learn)
- Hosted on: [Hugging Face Hub](https://huggingface.co/manthansubhash01/churn-prediction-model)
- Features: Total Spend, Support Calls, Payment Delay, Contract Length

## Project Structure

```
Customer-Churn-Prediction/
├── app.py                       # Streamlit entry point
├── requirements.txt
├── runtime.txt                  # python-3.11
├── packages.txt                 # empty (no apt deps)
├── .streamlit/secrets.toml      # local only, gitignored; holds GROQ_API_KEY
├── EXPLANATION.md               # detailed walkthrough (viva notes)
├── src/
│   ├── config.py
│   ├── model_handler.py         # loads RandomForest from HF
│   ├── data_processor.py
│   ├── predictor.py             # predict_single / predict_batch
│   ├── ui_components.py
│   ├── agent/                   # LangGraph agent
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── llm.py               # Groq client + chat_json
│   │   ├── prompts.py
│   │   ├── nodes.py             # classify_risk, build_query, retrieve, generate, compose, fallback
│   │   └── graph.py             # compiled StateGraph + run_agent
│   ├── rag/                     # FAISS retrieval
│   │   ├── corpus/              # 6 curated markdown playbooks
│   │   ├── index_store/         # faiss.index + meta.pkl (committed)
│   │   ├── index.py             # build / load index
│   │   └── retriever.py         # cached search(query, k)
│   ├── report/
│   │   └── pdf.py               # ReportLab PDF builder
│   └── dashboard/
│       └── analytics.py         # Plotly dashboard renderer
├── tests/
│   └── smoke_agent.py
└── report_latex/
    ├── main.tex
    └── references.bib
```

## Tech Stack

- **ML**: scikit-learn, pandas, numpy
- **LLM**: Groq Llama 3.3 70B (free tier)
- **Agent**: LangGraph (explicit state machine + conditional fallback)
- **RAG**: FAISS + sentence-transformers (`all-MiniLM-L6-v2`)
- **UI**: Streamlit
- **PDF**: ReportLab
- **Charts**: Plotly
- **Hosting**: Streamlit Community Cloud (free)
