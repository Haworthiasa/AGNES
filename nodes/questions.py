"""Generate discussion questions from paper analysis."""

import json
from typing import Any

from langchain_core.messages import HumanMessage

from llm import get_llm
from prompts import questions_prompt
from state import PaperState


def _content_to_text(content: Any) -> str:
    """Convert LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    return str(content)


def generate_questions(state: PaperState) -> dict[str, str]:
    """Generate markdown discussion questions from paper analysis."""
    analysis: dict = state["analysis"]
    if not analysis:
        error_message: str = "Error: analysis is empty"
        print(error_message)
        return {"discussion_questions": error_message}

    print("Generating discussion questions")

    try:
        analysis_json: str = json.dumps(analysis, indent=2)
        prompt: str = questions_prompt.format(analysis_json=analysis_json)
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        discussion_questions: str = _content_to_text(response.content).strip()

        if not discussion_questions:
            error_message = "Error: LLM returned empty discussion questions"
            print(error_message)
            return {"discussion_questions": error_message}

        print("Discussion questions generated")
        return {"discussion_questions": discussion_questions}
    except Exception as exc:
        error_message = f"Error generating discussion questions: {exc}"
        print(error_message)
        return {"discussion_questions": error_message}
