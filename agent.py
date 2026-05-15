"""Main LangGraph entry point for the paper-to-prototype agent."""

import os
import sys

from langgraph.graph import END, START, StateGraph

from config import MAX_FIX_ATTEMPTS, OUTPUT_DIR
from nodes.analyzer import analyze_paper
from nodes.demo_gen import generate_demo
from nodes.demo_runner import fix_demo, verify_demo
from nodes.fetcher import fetch_paper
from nodes.questions import generate_questions
from nodes.saver import save_results
from state import PaperState


def route_after_verify(state: PaperState) -> str:
    """Route to save or fix based on demo verification state."""
    if (
        state["demo_status"] == "passed"
        or state["fix_attempts"] >= MAX_FIX_ATTEMPTS
    ):
        return "save_results"
    return "fix_demo"


builder: StateGraph = StateGraph(PaperState)
builder.add_node("fetch_paper", fetch_paper)
builder.add_node("analyze_paper", analyze_paper)
builder.add_node("generate_questions", generate_questions)
builder.add_node("generate_demo", generate_demo)
builder.add_node("verify_demo", verify_demo)
builder.add_node("fix_demo", fix_demo)
builder.add_node("save_results", save_results)

builder.add_edge(START, "fetch_paper")
builder.add_edge("fetch_paper", "analyze_paper")
builder.add_edge("analyze_paper", "generate_questions")
builder.add_edge("analyze_paper", "generate_demo")
builder.add_edge(["generate_questions", "generate_demo"], "verify_demo")
builder.add_conditional_edges(
    "verify_demo",
    route_after_verify,
    {
        "save_results": "save_results",
        "fix_demo": "fix_demo",
    },
)
builder.add_edge("fix_demo", "verify_demo")
builder.add_edge("save_results", END)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python agent.py <arxiv_url>")
        sys.exit(1)

    url: str = sys.argv[1]
    initial_state: PaperState = {
        "arxiv_url": url,
        "paper_content": "",
        "paper_title": "",
        "analysis": {},
        "discussion_questions": "",
        "code_demo": "",
        "code_path": "",
        "demo_status": "pending",
        "fix_attempts": 0,
        "output_path": "",
    }

    graph = builder.compile()
    result = graph.invoke(initial_state)
    print(f"\nDone! Results saved to: {result.get('output_path', 'N/A')}")
