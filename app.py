import argparse
import operator
import uuid
import mlflow
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define shared state
class AgentState(TypedDict):
    question: str
    research: str
    analysis: str
    final_answer: str
    # Reducer: new items are appended instead of overwriting.
    # Without a checkpointer this resets every question;
    # with a checkpointer it grows across the conversation.
    history: Annotated[list[str], operator.add]


# 2. Research agent
def research_agent(state: AgentState):
    print("\n--- Research Agent ---")
    research = (
        "LangGraph is a framework for building "
        "stateful AI workflows."
    )
    print("Question:", state["question"])
    print("Research:", research)
    # Record this question in the conversation history
    return {"research": research, "history": [state["question"]]}


# 3. Analysis agent
def analysis_agent(state: AgentState):
    print("\n--- Analysis Agent ---")
    # Read information written by Research Agent
    research = state["research"]
    analysis = (
        f"Analysis based on research: {research} "
        "It supports graph-based workflow orchestration."
    )
    print("Input research:", research)
    print("Analysis:", analysis)
    return {"analysis": analysis}


# 4. Final answer agent
def answer_agent(state: AgentState):
    print("\n--- Answer Agent ---")
    # Everything except the current question
    previous = state["history"][:-1]
    answer = (
        f"Question: {state['question']}\n"
        f"Previous questions remembered: {previous or 'none'}\n"
        f"Research: {state['research']}\n"
        f"Analysis: {state['analysis']}\n"
        "Final Answer: LangGraph helps build "
        "stateful AI applications."
    )
    print(answer)
    return {"final_answer": answer}


# 5. Build LangGraph
builder = StateGraph(AgentState)
builder.add_node("research_agent", research_agent)
builder.add_node("analysis_agent", analysis_agent)
builder.add_node("answer_agent", answer_agent)
builder.add_edge(START, "research_agent")
builder.add_edge("research_agent", "analysis_agent")
builder.add_edge("analysis_agent", "answer_agent")
builder.add_edge("answer_agent", END)

# Same graph compiled two ways:
# - state only: state lives for one invocation, then is gone
# - checkpointer: state is saved per thread_id and reloaded next time
graph = builder.compile()
memory_graph = builder.compile(checkpointer=InMemorySaver())

# 6. Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LangGraph-State-Demo")
# Trace every graph execution: one span per agent node,
# with inputs, outputs and timings in the Traces tab
mlflow.langchain.autolog()


# 7. Run the graph for one question and log it as an MLflow run
def run_question(question: str, use_memory: bool, thread_id: str, turn: int):
    initial_state = {
        "question": question,
        "research": "",
        "analysis": "",
        "final_answer": "",
        "history": []
    }
    mode = "checkpointer" if use_memory else "state"
    app = memory_graph if use_memory else graph
    config = {"configurable": {"thread_id": thread_id}} if use_memory else None

    with mlflow.start_run(run_name=f"Multi-Agent-Execution-{mode}"):
        mlflow.log_params({
            "question": question[:500],
            "mode": mode,
            "thread_id": thread_id if use_memory else "none",
            "turn": turn
        })

        state_updates = []
        final_state = initial_state
        # One pass: "updates" gives each node's output,
        # "values" gives the full state after each node
        for stream_mode, chunk in app.stream(
            initial_state,
            config,
            stream_mode=["updates", "values"]
        ):
            if stream_mode == "updates":
                print("\nState update:", chunk)
                state_updates.append(chunk)
            else:
                final_state = chunk

        print("\n===== FINAL STATE =====")
        print(final_state)
        mlflow.log_metric("history_length", len(final_state["history"]))

        # Save final state in MLflow artifacts
        mlflow.log_dict(
            final_state,
            "final_state.json"
        )

        # Save updates from individual nodes
        mlflow.log_dict(
            state_updates,
            "state_updates.json"
        )

        # Save every checkpoint the checkpointer holds for this thread
        if use_memory:
            checkpoints = [
                {
                    "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                    "step": snap.metadata.get("step"),
                    "next": list(snap.next),
                    "created_at": snap.created_at,
                    "values": snap.values
                }
                for snap in memory_graph.get_state_history(config)
            ]
            mlflow.log_dict(checkpoints, "checkpoints.json")
            print(f"\nCheckpointer holds {len(checkpoints)} checkpoints "
                  f"for thread {thread_id}.")

        print("\nState information saved to MLflow.")


# 8. Ask questions from the terminal until the user exits
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use a checkpointer so state is remembered across questions"
    )
    args = parser.parse_args()

    thread_id = str(uuid.uuid4())[:8]
    turn = 0
    if args.memory:
        print(f"Mode: CHECKPOINTER (thread {thread_id}) - "
              "state is remembered across questions.")
        print("Type 'new' to start a new conversation thread.")
    else:
        print("Mode: STATE ONLY - every question starts from a fresh state.")
    print("Ask a question (type 'exit' or 'quit' to stop).")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        if args.memory and question.lower() == "new":
            thread_id = str(uuid.uuid4())[:8]
            turn = 0
            print(f"Started new thread {thread_id}.")
            continue
        turn += 1
        run_question(question, args.memory, thread_id, turn)
    print("\nGoodbye!")
