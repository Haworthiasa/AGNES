# Paper-to-Prototype Agent

## What This Project Does

This project is a LangGraph agent that turns an arXiv paper into a small set of practical artifacts:

- extracted paper text
- structured analysis
- discussion questions
- a runnable Python demo
- a verification/fix loop for the generated demo
- a final report saved to disk

The goal is to move from paper reading to a quick prototype that can be inspected, discussed, and run locally.

## Project Structure

```text
.
|-- agent.py              # Main LangGraph graph and CLI entry point
|-- state.py              # PaperState TypedDict shared by all nodes
|-- config.py             # LLM config, output path, max fix attempts
|-- llm.py                # ChatOpenAI factories for analysis and code
|-- prompts.py            # Prompts for analysis, questions, demo, and fixes
|-- requirements.txt      # Python dependencies
|-- AGENTS.md             # Project conventions
|-- nodes/
|   |-- fetcher.py        # Downloads paper PDF and extracts text
|   |-- analyzer.py       # Produces structured JSON analysis
|   |-- questions.py      # Generates markdown discussion questions
|   |-- demo_gen.py       # Generates Python demo code
|   |-- demo_runner.py    # Verifies and fixes generated demo code
|   `-- saver.py          # Saves final artifacts
`-- output/papers/        # Generated PDFs and output artifacts
```

Local folders such as `venv/`, `__pycache__/`, and generated files under `output/papers/` are runtime artifacts, not core source files.

## Components

- `PaperState`: the graph state shared across nodes. It stores the arXiv URL, paper content, title, analysis, questions, demo code/path, demo status, fix attempts, and final output path.
- `fetch_paper`: extracts the arXiv ID, downloads the PDF, and extracts text with `pypdf`.
- `analyze_paper`: calls the LLM and returns structured analysis with fields such as `problem`, `architecture`, `math`, `results`, and `one_sentence`.
- `generate_questions`: creates markdown discussion questions from the analysis.
- `generate_demo`: generates a Python demo from the analysis and saves it to `output/papers/demo.py`.
- `verify_demo`: runs the generated demo with the current Python interpreter and sets `demo_status` to `passed` or `failed`.
- `fix_demo`: if verification fails and the max attempt limit has not been reached, asks the code LLM to repair the demo and writes the updated file.
- `save_results`: saves `analysis.json`, optional `discussion_questions.md`, optional `demo.py`, and `report.md` under `output/papers/<paper_id>/`.

## Workflow / Graphflow

The main graph is defined in `agent.py`:

```text
START
  -> fetch_paper
  -> analyze_paper
  -> generate_questions + generate_demo
  -> verify_demo
  -> save_results -> END
  -> fix_demo -> verify_demo
```

Detailed routing:

```text
START -> fetch -> analyze -> gen_questions --+
                           -> gen_demo ------+
                                             |
                                        verify_demo
                                      /          \
                                    passed        failed (fix_attempts < 3)
                                      |               |
                                      |          fix_demo -> verify_demo
                                      |               |
                                      |          failed (fix_attempts >= 3)
                                      v               v
                                        save -> END
```

The conditional route is implemented by `route_after_verify`:

- if `demo_status == "passed"`, save results
- if `fix_attempts >= MAX_FIX_ATTEMPTS`, save results even though the demo failed
- otherwise, run `fix_demo` and loop back to `verify_demo`

## How To Run

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create `.env` in the project root:

```env
OPENCODE_GO_API_KEY=your_api_key_here
```

4. Run the agent:

```powershell
python agent.py https://arxiv.org/abs/1409.0473
```

Results are saved under:

```text
output/papers/<arxiv_id>/
```

## Current Status

- The main graph is implemented in `agent.py`.
- All six workflow node modules exist under `nodes/`.
- The graph uses `verify_demo` and `fix_demo` directly; `run_demo` still exists in `nodes/demo_runner.py` as a compatibility wrapper.
- The project currently has no `tests/` directory.
- `config.py` and `.env.example` both use `OPENCODE_GO_API_KEY`.
- Dependency versions are not pinned in `requirements.txt`.
