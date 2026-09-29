"""
Multi-Agent System - Orchestrate multiple specialized agents with LangGraph.
Demonstrates a Researcher + Writer + Reviewer pipeline.

Usage:
    python -m agents.multi_agent --topic "The future of AI agents"
"""

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


# ── State Definition ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state between agents."""
    topic: str
    research: str
    draft: str
    review: str
    final_output: str
    iteration: int


# ── Agent Definitions ───────────────────────────────────────────────────────────

def get_llm(model: str = "gpt-4o-mini"):
    return ChatOpenAI(
        model=model,
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def researcher_agent(state: AgentState) -> AgentState:
    """Researches a topic and produces structured notes."""
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a thorough research analyst. Given a topic, 
produce comprehensive research notes including:
- Key concepts and definitions
- Current trends and developments  
- Important statistics or facts
- Different perspectives on the topic
- Potential implications

Be factual and well-organized. Use bullet points and headers."""),
        HumanMessage(content=f"Research this topic thoroughly: {state['topic']}"),
    ]
    response = llm.invoke(messages)
    state["research"] = response.content
    print("  ✅ Researcher finished")
    return state


def writer_agent(state: AgentState) -> AgentState:
    """Writes a polished article based on research notes."""
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a skilled technical writer. Using the provided 
research notes, write a polished, engaging article. Include:
- A compelling introduction
- Well-structured body with clear sections
- Concrete examples and analogies
- A thoughtful conclusion
- Professional tone but accessible language

Do NOT make up facts — only use information from the research notes."""),
        HumanMessage(content=f"Write an article about '{state['topic']}' using these research notes:\n\n{state['research']}"),
    ]
    response = llm.invoke(messages)
    state["draft"] = response.content
    print("  ✅ Writer finished")
    return state


def reviewer_agent(state: AgentState) -> AgentState:
    """Reviews the draft and provides feedback or approves it."""
    llm = get_llm(model="gpt-4o-mini")
    messages = [
        SystemMessage(content="""You are a senior editor. Review the article draft and either:
1. APPROVE it if it's high quality (respond with "APPROVED" followed by any minor suggestions)
2. Provide specific, actionable feedback for improvement

Evaluate on: accuracy, clarity, structure, engagement, and completeness.
Be constructive but rigorous."""),
        HumanMessage(content=f"Review this article draft:\n\n{state['draft']}"),
    ]
    response = llm.invoke(messages)
    state["review"] = response.content
    state["iteration"] = state.get("iteration", 0) + 1
    print(f"  ✅ Reviewer finished (iteration {state['iteration']})")
    return state


# ── Graph Builder ───────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Decide whether to revise or finish."""
    if "APPROVED" in state.get("review", "").upper() or state.get("iteration", 0) >= 2:
        return "finish"
    return "revise"


def build_multi_agent_graph():
    """Build the multi-agent pipeline using LangGraph."""
    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("researcher", researcher_agent)
        graph.add_node("writer", writer_agent)
        graph.add_node("reviewer", reviewer_agent)

        # Add edges
        graph.set_entry_point("researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", "reviewer")

        # Conditional: revise or finish
        graph.add_conditional_edges(
            "reviewer",
            should_continue,
            {
                "revise": "writer",  # Send back to writer with feedback
                "finish": END,
            },
        )

        return graph.compile()

    except ImportError:
        print("⚠️  LangGraph not installed. Install with: pip install langgraph")
        print("   Falling back to sequential execution...\n")
        return None


def run_sequential(topic: str) -> AgentState:
    """Fallback: run agents sequentially without LangGraph."""
    state: AgentState = {
        "topic": topic,
        "research": "",
        "draft": "",
        "review": "",
        "final_output": "",
        "iteration": 0,
    }

    state = researcher_agent(state)
    state = writer_agent(state)
    state = reviewer_agent(state)
    state["final_output"] = state["draft"]
    return state


# ── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Content Pipeline")
    parser.add_argument("--topic", "-t", type=str, required=True, help="Topic to research and write about")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="LLM model")
    args = parser.parse_args()

    print(f"🎯 Topic: {args.topic}\n")
    print("🚀 Running multi-agent pipeline...\n")

    graph = build_multi_agent_graph()

    if graph:
        initial_state: AgentState = {
            "topic": args.topic,
            "research": "",
            "draft": "",
            "review": "",
            "final_output": "",
            "iteration": 0,
        }
        result = graph.invoke(initial_state)
    else:
        result = run_sequential(args.topic)

    print("\n" + "=" * 70)
    print("📝 FINAL ARTICLE")
    print("=" * 70)
    print(result.get("draft", result.get("final_output", "No output")))
    print("\n" + "=" * 70)
    print("📋 REVIEWER NOTES")
    print("=" * 70)
    print(result.get("review", "No review"))


if __name__ == "__main__":
    main()
