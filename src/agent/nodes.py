from .llm import chat_json
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE, DISCLAIMER


HIGH = 0.7
MEDIUM = 0.4


def classify_risk(state):
    prob = state["prediction"]["churn_probability"]
    if prob >= HIGH:
        tier = "high"
    elif prob >= MEDIUM:
        tier = "medium"
    else:
        tier = "low"
    return {"risk_tier": tier}


def build_query(state):
    customer = state["customer"]
    reasons = state["prediction"].get("churn_reasons", [])
    top = ", ".join(r["reason"] for r in reasons[:3]) or "general churn risk"
    q = (
        f"{state['risk_tier']} churn risk, contract {customer['contract_length']}, "
        f"{top}, retention strategies"
    )
    return {"query": q}


def retrieve_context(state):
    try:
        from src.rag.retriever import search
        hits = search(state["query"], k=5)
        return {"retrieved": hits}
    except Exception as e:
        return {"retrieved": [], "error": f"retrieval_failed: {e}"}


def generate_strategy(state):
    customer = state["customer"]
    pred = state["prediction"]
    reasons = pred.get("churn_reasons", [])
    reasons_block = "\n".join(
        f"- {r['reason']} ({r['severity']}): {r['detail']}" for r in reasons
    ) or "- No dominant single factor."

    snippets = state.get("retrieved", [])
    snippets_block = "\n".join(
        f"[{i+1}] ({s['source']} - {s['heading']}) {s['text']}"
        for i, s in enumerate(snippets)
    ) or "No snippets retrieved."

    user_prompt = USER_TEMPLATE.format(
        total_spend=f"{customer['total_spend']:.0f}",
        support_calls=customer["support_calls"],
        payment_delay=customer["payment_delay"],
        contract_length=customer["contract_length"],
        churn_pct=f"{pred['churn_probability']*100:.1f}",
        risk_tier=state["risk_tier"],
        reasons_block=reasons_block,
        snippets_block=snippets_block,
    )

    output = chat_json(SYSTEM_PROMPT, user_prompt)
    if output is None:
        return {"error": "llm_failed"}

    required = {"risk_summary", "factors", "recommendations"}
    if not required.issubset(output.keys()):
        return {"error": "llm_bad_schema"}

    return {"report": output}


def compose_report(state):
    report = dict(state.get("report") or {})
    sources = []
    seen = set()
    for s in state.get("retrieved", []):
        key = (s["source"], s["heading"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": s["source"], "section": s["heading"]})
    report["sources"] = sources
    report["disclaimer"] = DISCLAIMER
    report["risk_tier"] = state["risk_tier"]
    report["churn_probability"] = state["prediction"]["churn_probability"]
    return {"report": report, "error": None}


def fallback_report(state):
    pred = state["prediction"]
    reasons = pred.get("churn_reasons", [])
    tier = state.get("risk_tier", "medium")

    factors = [
        {"name": r["reason"], "severity": r["severity"], "evidence": r["detail"]}
        for r in reasons
    ]

    tips = {
        "Very High Support Call Volume": ("Assign a retention specialist for a proactive call to resolve the root cause.", "immediate"),
        "Elevated Support Calls": ("Review open tickets and schedule a follow-up within 48 hours.", "short-term"),
        "Significant Payment Delays": ("Offer a flexible payment plan or autopay discount.", "immediate"),
        "Moderate Payment Delays": ("Send a friendly payment reminder and highlight autopay benefits.", "short-term"),
        "Very Low Spending": ("Share a personalised feature walkthrough to drive adoption.", "short-term"),
        "Below-Average Spending": ("Recommend relevant add-ons or higher-value plans.", "short-term"),
        "Month-to-Month Contract": ("Offer a discounted annual plan to lock in the customer.", "short-term"),
        "Short-Term Contract": ("Propose a longer contract with a loyalty discount.", "long-term"),
        "Combined Risk Factors": ("Trigger a general retention campaign with a personalised offer.", "short-term"),
    }

    recommendations = []
    for r in reasons[:5]:
        action, timeframe = tips.get(
            r["reason"],
            ("Reach out to the customer to understand their needs.", "short-term"),
        )
        recommendations.append({
            "action": action,
            "rationale": r["detail"],
            "expected_impact": "Reduced churn probability for this customer.",
            "timeframe": timeframe,
        })

    if not recommendations:
        recommendations.append({
            "action": "Maintain engagement with regular check-ins and loyalty perks.",
            "rationale": "Customer is low risk but should be monitored.",
            "expected_impact": "Continued retention.",
            "timeframe": "long-term",
        })

    sources = []
    seen = set()
    for s in state.get("retrieved", []):
        key = (s["source"], s["heading"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": s["source"], "section": s["heading"]})

    report = {
        "risk_summary": f"This customer is at {tier} risk of churn with a probability of {pred['churn_probability']*100:.1f}%.",
        "factors": factors,
        "recommendations": recommendations,
        "confidence_note": "Generated from rule-based fallback because the language model was unavailable.",
        "sources": sources,
        "disclaimer": DISCLAIMER,
        "risk_tier": tier,
        "churn_probability": pred["churn_probability"],
        "used_fallback": True,
    }

    return {"report": report}
