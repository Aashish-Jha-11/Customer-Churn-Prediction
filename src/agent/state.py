from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    customer: Dict[str, Any]
    prediction: Dict[str, Any]
    risk_tier: str
    query: str
    retrieved: List[Dict[str, Any]]
    report: Dict[str, Any]
    error: Optional[str]
