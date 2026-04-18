import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fake_chat(system, user):
    return {
        "risk_summary": "Test summary showing customer is at risk.",
        "factors": [
            {"name": "Payment Delay", "severity": "high", "evidence": "Delayed 25 days."}
        ],
        "recommendations": [
            {
                "action": "Offer autopay discount",
                "rationale": "Payment delay detected",
                "expected_impact": "Lower involuntary churn",
                "timeframe": "immediate",
            }
        ],
        "confidence_note": "Test run.",
    }


def fake_search(query, k=5):
    return [
        {
            "text": "Sample snippet.",
            "heading": "Sample Section",
            "doc": "Sample Doc",
            "source": "sample.md",
            "score": 0.9,
        }
    ]


def _run_case(churn_prob, reasons, label):
    from src.agent import graph as graph_module
    from src.agent import nodes as nodes_module

    graph_module._GRAPH = None

    customer = {
        "total_spend": 120,
        "support_calls": 5,
        "payment_delay": 25,
        "contract_length": "Monthly",
    }
    prediction = {
        "prediction": 1 if churn_prob > 0.5 else 0,
        "churn_probability": churn_prob,
        "retention_probability": 1 - churn_prob,
        "is_high_risk": churn_prob > 0.5,
        "churn_reasons": reasons,
    }

    with patch.object(nodes_module, "chat_json", side_effect=fake_chat), \
         patch("src.rag.retriever.search", side_effect=fake_search):
        from src.agent.graph import run_agent
        report = run_agent(customer, prediction)

    print(f"[{label}] tier={report['risk_tier']} "
          f"recs={len(report.get('recommendations', []))} "
          f"fallback={report.get('used_fallback', False)}")
    required = {"risk_summary", "factors", "recommendations", "sources", "disclaimer"}
    assert required.issubset(report.keys()), f"missing keys in {label}: {set(report) ^ required}"
    assert report["risk_tier"] == (
        "high" if churn_prob >= 0.7 else "medium" if churn_prob >= 0.4 else "low"
    )
    return report


def _run_fallback_case():
    from src.agent import graph as graph_module
    from src.agent import nodes as nodes_module

    graph_module._GRAPH = None

    customer = {
        "total_spend": 200,
        "support_calls": 8,
        "payment_delay": 30,
        "contract_length": "Monthly",
    }
    prediction = {
        "prediction": 1,
        "churn_probability": 0.85,
        "retention_probability": 0.15,
        "is_high_risk": True,
        "churn_reasons": [
            {"reason": "Very High Support Call Volume", "detail": "8 calls", "severity": "high"},
            {"reason": "Significant Payment Delays", "detail": "30 days", "severity": "high"},
        ],
    }

    with patch.object(nodes_module, "chat_json", return_value=None), \
         patch("src.rag.retriever.search", side_effect=fake_search):
        from src.agent.graph import run_agent
        report = run_agent(customer, prediction)

    print(f"[fallback] used_fallback={report.get('used_fallback')} "
          f"recs={len(report['recommendations'])}")
    assert report.get("used_fallback") is True
    assert len(report["recommendations"]) >= 1


if __name__ == "__main__":
    high_reasons = [
        {"reason": "Significant Payment Delays", "detail": "25 days", "severity": "high"},
        {"reason": "Month-to-Month Contract", "detail": "Easy to leave", "severity": "medium"},
    ]
    medium_reasons = [
        {"reason": "Moderate Payment Delays", "detail": "12 days", "severity": "medium"},
    ]
    low_reasons = []

    _run_case(0.85, high_reasons, "high")
    _run_case(0.55, medium_reasons, "medium")
    _run_case(0.2, low_reasons, "low")
    _run_fallback_case()
    print("All smoke tests passed.")
