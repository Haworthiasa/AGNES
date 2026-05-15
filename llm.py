"""LangChain LLM factories."""

from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def get_llm() -> ChatOpenAI:
    """Return the default analysis LLM."""
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.2,
    )


def get_code_llm() -> ChatOpenAI:
    """Return the code generation and repair LLM."""
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.1,
    )
