"""Generate and save a runnable paper prototype demo."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import OUTPUT_DIR
from llm import get_code_llm
from prompts import demo_prompt
from state import PaperState


def _content_to_text(content: Any) -> str:
    """Convert LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    return str(content)


def _strip_code_fences(code: str) -> str:
    """Remove markdown fences if the LLM includes them."""
    stripped_code: str = code.strip()
    if not stripped_code.startswith("```"):
        return stripped_code

    lines: list[str] = stripped_code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def generate_demo(state: PaperState) -> dict[str, str]:
    """Generate Python demo code from paper analysis and save it to disk."""
    analysis: dict = state["analysis"]
    if not analysis:
        error_message: str = "Error: analysis is empty"
        print(error_message)
        return {
            "code_demo": error_message,
            "code_path": "",
        }

    print("Generating code demo")

    try:
        analysis_json: str = json.dumps(analysis, indent=2)
        user_prompt: str = (
            "Generate a Python demo from this paper analysis and architecture "
            f"description:\n\n{analysis_json}"
        )
        llm = get_code_llm()
        response = llm.invoke(
            [
                SystemMessage(content=demo_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        code_demo: str = _strip_code_fences(_content_to_text(response.content))

        if not code_demo:
            error_message = "Error: LLM returned empty demo code"
            print(error_message)
            return {
                "code_demo": error_message,
                "code_path": "",
            }

        output_dir: Path = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        code_path: Path = output_dir / "demo.py"
        code_path.write_text(code_demo, encoding="utf-8")

        print(f"Code demo saved to {code_path}")
        return {
            "code_demo": code_demo,
            "code_path": str(code_path),
        }
    except Exception as exc:
        error_message = f"Error generating demo: {exc}"
        print(error_message)
        return {
            "code_demo": error_message,
            "code_path": "",
        }
