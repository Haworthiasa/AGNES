"""Shared LangGraph state for the paper-to-prototype workflow."""

from typing import TypedDict


class PaperState(TypedDict):
    """State passed between paper agent nodes."""

    arxiv_url: str
    paper_content: str
    paper_title: str
    analysis: dict
    discussion_questions: str
    code_demo: str
    code_path: str
    demo_status: str
    fix_attempts: int
    output_path: str
