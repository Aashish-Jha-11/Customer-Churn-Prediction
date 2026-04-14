from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    classify_risk,
    build_query,
    retrieve_context,
    generate_strategy,
    compose_report,
    fallback_report,
)


def _route_after_retrieve(state):
    if state.get("error"):
        return "fallback"
    return "generate"


def _route_after_generate(state):
    if state.get("error"):
        return "fallback"
    return "compose"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify_risk", classify_risk)
    g.add_node("build_query", build_query)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("generate_strategy", generate_strategy)
    g.add_node("compose_report", compose_report)
    g.add_node("fallback_report", fallback_report)

    g.set_entry_point("classify_risk")
    g.add_edge("classify_risk", "build_query")
    g.add_edge("build_query", "retrieve_context")
    g.add_conditional_edges(
        "retrieve_context",
        _route_after_retrieve,
        {"fallback": "fallback_report", "generate": "generate_strategy"},
    )
    g.add_conditional_edges(
        "generate_strategy",
        _route_after_generate,
        {"fallback": "fallback_report", "compose": "compose_report"},
    )
    g.add_edge("compose_report", END)
    g.add_edge("fallback_report", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(customer, prediction):
    graph = get_graph()
    initial = {"customer": customer, "prediction": prediction}
    final = graph.invoke(initial)
    return final["report"]
