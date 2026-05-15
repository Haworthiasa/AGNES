"""Project configuration for the paper-to-prototype agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT_DIR: Path = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT_DIR / ".env")

LLM_API_KEY: str = os.getenv("OPENCODE_GO_API_KEY", "")
LLM_BASE_URL: str = "https://opencode.ai/zen/go/v1"
LLM_MODEL: str = "deepseek-v4-flash"

OUTPUT_DIR: str = "output/papers"
MAX_FIX_ATTEMPTS: int = 3
