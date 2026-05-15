"""Analyze extracted paper content with an LLM."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from prompts import analyzer_prompt
from state import PaperState


ANALYSIS_KEYS: tuple[str, ...] = (
    "problem",
    "architecture",
    "math",
    "results",
    "one_sentence",
)


def _error_analysis(message: str) -> dict[str, str]:
    """Return an analysis payload that preserves the expected keys."""
    return {
        "problem": message,
        "architecture": "",
        "math": "",
        "results": "",
        "one_sentence": message,
    }


def _content_to_text(content: Any) -> str:
    """Convert LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    return str(content)


def _parse_analysis(raw_response: str) -> dict[str, str]:
    """Parse and normalize the LLM JSON response."""
    parsed: Any = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")

    analysis: dict[str, str] = {}
    for key in ANALYSIS_KEYS:
        value: Any = parsed.get(key, "")
        analysis[key] = value if isinstance(value, str) else json.dumps(value)
    return analysis


def analyze_paper(state: PaperState) -> dict[str, dict[str, str]]:
    """Analyze paper content and return structured analysis fields."""
    paper_content: str = state["paper_content"].strip()
    if not paper_content:
        error_message: str = "Error: paper_content is empty"
        print(error_message)
        return {"analysis": _error_analysis(error_message)}
    if paper_content.startswith("Error:"):
        print(f"Skipping analysis because fetch failed: {paper_content}")
        return {"analysis": _error_analysis(paper_content)}

    print("Analyzing paper content")

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=analyzer_prompt),
            HumanMessage(content=paper_content),
        ]
        response = llm.invoke(messages)
        raw_response: str = _content_to_text(response.content).strip()
        try:
            analysis: dict[str, str] = _parse_analysis(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Invalid analysis JSON, retrying once: {exc}")
            retry_messages = [
                SystemMessage(content=analyzer_prompt),
                HumanMessage(content=paper_content),
                HumanMessage(content="Return valid JSON please"),
            ]
            retry_response = llm.invoke(retry_messages)
            retry_raw_response: str = _content_to_text(retry_response.content).strip()
            analysis = _parse_analysis(retry_raw_response)

        print("Paper analysis complete")
        return {"analysis": analysis}
    except Exception as exc:
        error_message = f"Error analyzing paper: {exc}"
        print(error_message)
        return {"analysis": _error_analysis(error_message)}
