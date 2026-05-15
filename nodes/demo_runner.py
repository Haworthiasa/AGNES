"""Run generated demo code and repair failures."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import MAX_FIX_ATTEMPTS, OUTPUT_DIR
from llm import get_code_llm
from prompts import fix_prompt
from state import PaperState


RUN_TIMEOUT_SECONDS: int = 30


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


def _get_code_file(state: PaperState) -> tuple[Path | None, str]:
    """Resolve demo code to a runnable file path and code string."""
    code_path_value: str = state.get("code_path", "")
    if code_path_value:
        code_path = Path(code_path_value)
        if code_path.is_file():
            return code_path, code_path.read_text(encoding="utf-8")

    code_demo: str = state.get("code_demo", "").strip()
    if not code_demo:
        return None, ""

    output_dir: Path = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / "demo.py"
    code_path.write_text(code_demo, encoding="utf-8")
    return code_path, code_demo


def _run_code(code_path: Path) -> tuple[bool, str]:
    """Run the generated demo file and return success plus process output."""
    try:
        result = subprocess.run(
            [sys.executable, code_path.name],
            cwd=code_path.parent,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout: str = exc.stdout or ""
        stderr: str = exc.stderr or ""
        output: str = "\n".join(part for part in [stdout, stderr] if part).strip()
        return False, f"Demo timed out after {RUN_TIMEOUT_SECONDS}s\n{output}".strip()

    output = "\n".join(
        part for part in [result.stdout, result.stderr] if part
    ).strip()
    if result.returncode == 0:
        return True, output
    return False, f"Exit code {result.returncode}\n{output}".strip()


def _fix_code(code: str, error_message: str) -> str:
    """Use the code LLM to repair failed demo code."""
    llm = get_code_llm()
    user_prompt: str = (
        "Fix this Python demo code using the error message.\n\n"
        f"Error message:\n{error_message}\n\n"
        f"Code:\n{code}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=fix_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return _strip_code_fences(_content_to_text(response.content))


def verify_demo(state: PaperState) -> dict[str, str | int]:
    """Run demo code once and return pass/fail status."""
    code_path, current_code = _get_code_file(state)
    fix_attempts: int = int(state.get("fix_attempts", 0) or 0)

    if code_path is None:
        error_message: str = "Error: no demo code available to run"
        print(error_message)
        return {
            "code_demo": error_message,
            "code_path": "",
            "demo_status": "failed",
            "fix_attempts": fix_attempts,
        }

    print(f"Running demo code: {code_path}")
    passed, output = _run_code(code_path)
    if passed:
        print("Demo passed")
        return {
            "code_demo": current_code,
            "code_path": str(code_path),
            "demo_status": "passed",
            "fix_attempts": fix_attempts,
        }

    print(f"Demo failed: {output}")
    return {
        "code_demo": current_code,
        "code_path": str(code_path),
        "demo_status": "failed",
        "fix_attempts": fix_attempts,
    }


def fix_demo(state: PaperState) -> dict[str, str | int]:
    """Repair demo code once using the latest runtime error."""
    code_path, current_code = _get_code_file(state)
    fix_attempts: int = int(state.get("fix_attempts", 0) or 0)

    if code_path is None:
        error_message: str = "Error: no demo code available to fix"
        print(error_message)
        return {
            "code_demo": error_message,
            "code_path": "",
            "demo_status": "failed",
            "fix_attempts": fix_attempts,
        }

    if fix_attempts >= MAX_FIX_ATTEMPTS:
        print("Maximum fix attempts reached")
        return {
            "code_demo": current_code,
            "code_path": str(code_path),
            "demo_status": "failed",
            "fix_attempts": fix_attempts,
        }

    print(f"Collecting demo error before fix: {code_path}")
    passed, output = _run_code(code_path)
    if passed:
        print("Demo already passes")
        return {
            "code_demo": current_code,
            "code_path": str(code_path),
            "demo_status": "passed",
            "fix_attempts": fix_attempts,
        }

    fix_attempts += 1
    print(f"Fix attempt {fix_attempts}/{MAX_FIX_ATTEMPTS}")

    try:
        fixed_code: str = _fix_code(current_code, output)
    except Exception as exc:
        error_message = f"Error fixing demo code: {exc}"
        print(error_message)
        return {
            "code_demo": current_code,
            "code_path": str(code_path),
            "demo_status": "failed",
            "fix_attempts": fix_attempts,
        }

    if not fixed_code:
        print("Error fixing demo code: LLM returned empty code")
        return {
            "code_demo": current_code,
            "code_path": str(code_path),
            "demo_status": "failed",
            "fix_attempts": fix_attempts,
        }

    code_path.write_text(fixed_code, encoding="utf-8")
    print(f"Fixed demo code saved to {code_path}")
    return {
        "code_demo": fixed_code,
        "code_path": str(code_path),
        "demo_status": "pending",
        "fix_attempts": fix_attempts,
    }


def run_demo(state: PaperState) -> dict[str, str | int]:
    """Run demo code and auto-fix failures up to MAX_FIX_ATTEMPTS."""
    result: dict[str, str | int] = verify_demo(state)
    current_state: PaperState = {**state, **result}

    while (
        current_state["demo_status"] != "passed"
        and current_state["fix_attempts"] < MAX_FIX_ATTEMPTS
    ):
        fix_result: dict[str, str | int] = fix_demo(current_state)
        current_state = {**current_state, **fix_result}
        result = verify_demo(current_state)
        current_state = {**current_state, **result}

    return {
        "code_demo": current_state["code_demo"],
        "code_path": current_state["code_path"],
        "demo_status": current_state["demo_status"],
        "fix_attempts": current_state["fix_attempts"],
    }
