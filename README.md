# LangGraph + Checkpointer + MLflow Tracing

A small multi-agent [LangGraph](https://langchain-ai.github.io/langgraph/) app that shows how agents share state, how a checkpointer remembers state across questions, and how to inspect all of it in [MLflow](https://mlflow.org/).

Three agents run in order and pass information through one shared state:

```
question -> research_agent -> analysis_agent -> answer_agent -> final answer
```

Every question you type in the terminal becomes an MLflow run with:

- **Artifacts**: `final_state.json`, `state_updates.json` and, in memory mode, `checkpoints.json`
- **A trace**: one span per agent, with inputs, outputs and timings (via `mlflow.langchain.autolog()`)
- **Params**: `question`, `mode`, `thread_id`, `turn`, plus a `history_length` metric

The agents return fixed text rather than calling an LLM, so the focus stays on state and tracking.

## Project structure

```
langgraph_mlflow_demo/
├── app.py            # graph, agents, MLflow logging, terminal loop
├── requirements.txt  # langgraph, mlflow
└── README.md
```

## Requirements

- Python 3.10+
- Windows PowerShell commands are shown; macOS/Linux equivalents are noted where they differ

## Setup (once)

```powershell
cd langgraph_mlflow_demo
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled on this system"*, run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run it

You need two terminals: one for the MLflow server and one for the app.

### Terminal 1: start the MLflow server

```powershell
.venv\Scripts\Activate.ps1
mlflow server --host 127.0.0.1 --port 5000
```

Keep it running and open <http://127.0.0.1:5000> in your browser.

### Terminal 2: run the app

```powershell
.venv\Scripts\Activate.ps1
python app.py            # state only: every question starts from a fresh state
python app.py --memory   # checkpointer: questions are remembered within a thread
```

Type a question and press Enter. Type `exit` or `quit` to stop. In `--memory` mode, type `new` to start a new conversation thread.

## Sample questions

Ask these in order, in one session:

1. `What is LangGraph?`
2. `How does it share data between agents?`
3. `What did I ask you before?`

| | `python app.py` | `python app.py --memory` |
|---|---|---|
| Q1 | Previous questions remembered: none | none |
| Q2 | none | `['What is LangGraph?']` |
| Q3 | none | `['What is LangGraph?', 'How does it share data between agents?']` |

## Inspect results in the MLflow UI

1. Open <http://127.0.0.1:5000> and select the experiment **LangGraph-State-Demo**.
2. **Runs**: one per question, named `Multi-Agent-Execution-state` or `Multi-Agent-Execution-checkpointer`. Add the `question`, `mode` and `turn` columns to tell them apart.
3. **Artifacts** tab of a run:

   | File | Contents |
   |---|---|
   | `final_state.json` | The full state after all agents ran |
   | `state_updates.json` | What each agent returned, in order |
   | `checkpoints.json` | Memory mode only: every saved snapshot for the thread |

4. **Traces** tab: one trace per question, with a span for each agent:

   ```
   LangGraph
   ├── research_agent
   ├── analysis_agent
   └── answer_agent
   ```

## State vs checkpointer

- **State** is the shared memory between agents during one run. Each agent reads it and returns only the fields it changes; LangGraph merges them.
- **A checkpointer** (`InMemorySaver`) saves the state after every step under a `thread_id`, so the next question on the same thread starts from the saved state. It writes 5 checkpoints per question: input, start, and one after each agent.

`InMemorySaver` keeps memory only while the app is running.

## Troubleshooting

| Problem | Fix |
|---|---|
| Connection refused when running `app.py` | Start the MLflow server first (Terminal 1) |
| Port 5000 already in use | A server is already running; use it, or stop it first |
| No traces in the Traces tab | Only runs made after tracing was added have traces; ask a new question and refresh |
| `ModuleNotFoundError` | Activate the virtual environment (`(.venv)` should show in the prompt) |

## Related

[`../langgraph_studio_demo`](../langgraph_studio_demo) runs the same graph in LangGraph Studio with LangSmith tracing.
