# Customer Churn Prediction and Retention Assistant

A Streamlit app that predicts whether a customer will churn and also generates
a short retention plan for each customer. The churn prediction uses a Random
Forest model trained on the Kaggle customer churn dataset. The retention plan
is produced by a small agent that retrieves relevant notes from a local
knowledge base and asks the Groq Llama 3.3 70B model (free tier) to write the
final report.

## What you can do in the app

There are four modes in the sidebar:

1. Single Prediction: fill the form, get churn probability and the top reasons.
2. Batch Prediction: upload a CSV, download predictions as CSV.
3. Retention Report: fill the form, get an AI generated retention plan with
   sources, and download it as a PDF.
4. Analytics Dashboard: upload a CSV and see churn rate, risk tiers, a spend
   vs support scatter and a value vs risk segmentation.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Groq API key (for the Retention Report)

The Retention Report mode uses the Groq free tier. Get a key from
https://console.groq.com and create a file at `.streamlit/secrets.toml` with:

```toml
GROQ_API_KEY = "gsk_your_key"
```

If no key is set, the app still works — it falls back to a simple rule based
report built from the model's own reasons.

## CSV format for Batch and Dashboard modes

```csv
Total Spend,Support Calls,Payment Delay,Contract Length
500.0,2,0,Monthly
1200.0,5,15,Annual
800.0,1,5,Quarterly
```

Contract Length must be Monthly, Quarterly or Annual.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (already done).
2. Go to https://share.streamlit.io, click New app, pick this repo and
   `app.py` on the `main` branch.
3. Open Advanced settings, go to Secrets and paste
   `GROQ_API_KEY = "gsk_your_key"`.
4. Click Deploy. The first build takes a few minutes because it has to
   download sentence-transformers, FAISS and torch.

`runtime.txt` pins Python 3.11 so the FAISS wheel installs cleanly.
`packages.txt` is empty because we do not need any apt packages.

## Smoke test

A small test that runs the agent end-to-end with a mocked LLM and retriever.
No API calls, no network.

```bash
python -m tests.smoke_agent
```

## Model

- Random Forest classifier (scikit-learn), trained on the Kaggle telco churn
  dataset.
- Hosted on Hugging Face Hub:
  https://huggingface.co/manthansubhash01/churn-prediction-model
- Inputs: Total Spend, Support Calls, Payment Delay, Contract Length.

## Folder layout

```
Customer-Churn-Prediction/
├── app.py
├── requirements.txt
├── runtime.txt
├── packages.txt
├── src/
│   ├── config.py
│   ├── model_handler.py
│   ├── data_processor.py
│   ├── predictor.py
│   ├── ui_components.py
│   ├── agent/         # LangGraph agent (state, nodes, graph, Groq client)
│   ├── rag/           # corpus + FAISS index + retriever
│   ├── report/        # PDF builder (reportlab)
│   └── dashboard/     # batch analytics (plotly)
└── tests/
    └── smoke_agent.py
```

## Main libraries

scikit-learn, pandas, numpy, streamlit, groq, langgraph, langchain-core,
sentence-transformers, faiss-cpu, plotly, reportlab, huggingface-hub.
