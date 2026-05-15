# Paper-to-Prototype Agent

## Architecture
- LangGraph StateGraph: 1 state + 6 nodes (fetch → analyze → gen_questions → gen_demo → verify → save)
- State management: TypedDict
`PaperState`
- LLM: OpenAI-compatible API via langchain-openai (ChatOpenAI)
  - base_url: https://opencode.ai/zen/go/v1
  - model: deepseek-v4-flash
- Fetch paper: arxiv Python library + requests
- Run code: subprocess

## Tech Stack
- Python 3.13
- langchain, langchain-community, langchain-openai
- langgraph == latest
- langsmith (tracing)
- arxiv, pypdf

## Conventions
- Snake case for all Python (functions, vars, files)
- Each node in its own file under nodes/
- Type hints everywhere
- main entry: agent.py with `if __name__ == "__main__"`
- Output saved to output/papers/ as markdown + code files

## Commands
- Install deps: `pip install -r requirements.txt`
- Run: `python agent.py <arxiv_url>`
- Test: `python -m pytest tests/`

## Project Structure
paper-agent/
├── agent.py              # LangGraph graph definition
├── state.py              # PaperState TypedDict
├── config.py             # Config (model, paths, API keys)
├── llm.py                # LangChain LLM setup
├── prompts.py            # System prompts for each node
├── requirements.txt
├── AGENTS.md
├── nodes/
│   ├── __init__.py
│   ├── fetcher.py
│   ├── analyzer.py
│   ├── questions.py
│   ├── demo_gen.py
│   ├── demo_runner.py
│   └── saver.py
└── output/papers/