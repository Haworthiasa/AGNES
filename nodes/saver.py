"""Save paper analysis, questions, demo code, and report files."""

import json
import re
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from nodes.fetcher import extract_arxiv_id
from state import PaperState


def _safe_paper_id(arxiv_url: str) -> str:
    """Return a filesystem-safe paper ID for output directory names."""
    arxiv_id: str | None = extract_arxiv_id(arxiv_url)
    if arxiv_id is None:
        arxiv_id = "paper"

    safe_id: str = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id).strip("._-")
    return safe_id or "paper"


def _analysis_value(analysis: dict[str, Any], key: str) -> str:
    """Return a markdown-safe string from the analysis dict."""
    value: Any = analysis.get(key, "")
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, indent=2)


def _demo_code_from_state(state: PaperState) -> str:
    """Return demo code from state or an existing code_path."""
    code_demo: str = state.get("code_demo", "").strip()
    if code_demo:
        return code_demo

    code_path_value: str = state.get("code_path", "")
    if not code_path_value:
        return ""

    code_path = Path(code_path_value)
    if not code_path.is_file():
        return ""
    return code_path.read_text(encoding="utf-8").strip()


def generate_report(state: PaperState) -> str:
    """Generate a markdown report for the completed paper workflow."""
    analysis: dict[str, Any] = state.get("analysis", {})
    title: str = state.get("paper_title", "").strip() or "Untitled paper"
    arxiv_url: str = state.get("arxiv_url", "").strip()
    one_sentence: str = _analysis_value(analysis, "one_sentence")
    architecture: str = _analysis_value(analysis, "architecture")
    results: str = _analysis_value(analysis, "results")
    questions: str = state.get("discussion_questions", "").strip()
    demo_status: str = state.get("demo_status", "").strip() or "pending"

    return f"""# {title}

## URL
{arxiv_url or "Not provided"}

## One-Sentence
{one_sentence or "Not available"}

## Architecture
{architecture or "Not available"}

## Results
{results or "Not available"}

## Questions
{questions or "No discussion questions generated."}

## Demo Status
{demo_status}
""".strip()


def save_results(state: PaperState) -> dict[str, str]:
    """Save all generated paper artifacts and return the output directory."""
    paper_id: str = _safe_paper_id(state.get("arxiv_url", ""))
    output_path: Path = Path(OUTPUT_DIR) / paper_id
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {output_path}")

    analysis: dict = state.get("analysis", {})
    analysis_path: Path = output_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )

    discussion_questions: str = state.get("discussion_questions", "").strip()
    if discussion_questions:
        questions_path: Path = output_path / "discussion_questions.md"
        questions_path.write_text(discussion_questions, encoding="utf-8")

    demo_code: str = _demo_code_from_state(state)
    if demo_code:
        demo_path: Path = output_path / "demo.py"
        demo_path.write_text(demo_code, encoding="utf-8")

    report_path: Path = output_path / "report.md"
    report_path.write_text(generate_report(state), encoding="utf-8")

    print("Results saved")
    return {"output_path": str(output_path)}
