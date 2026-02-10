"""
Quickstart - Minimal example to get started with AI Agent Toolkit.

Usage:
    python examples/quickstart.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def demo_rag():
    """Quick RAG agent demo."""
    from langchain_core.documents import Document
    from agents.rag_agent import build_rag_chain

    docs = [
        Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"source": "facts"}),
        Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "facts"}),
        Document(page_content="RAG combines retrieval with generation for grounded AI.", metadata={"source": "facts"}),
    ]

    chain = build_rag_chain(documents=docs)
    answer = chain.invoke("What is LangChain?")
    print(f"RAG Answer: {answer}")


def demo_tool_agent():
    """Quick tool-calling agent demo."""
    from agents.tool_calling_agent import run_agent_loop

    answer = run_agent_loop("What is the square root of 144 plus 25?")
    print(f"Tool Agent Answer: {answer}")


def demo_memory_agent():
    """Quick memory agent demo."""
    from agents.memory_agent import MemoryAgent

    agent = MemoryAgent()
    print("Memory Agent:")
    print(f"  Q1: {agent.chat('My name is Manoj')}")
    print(f"  Q2: {agent.chat('What is my name?')}")


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 AI Agent Toolkit — Quickstart")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Set OPENAI_API_KEY in .env file first!")
        print("   cp .env.example .env")
        print("   # Edit .env and add your key")
        exit(1)

    print("\n--- Demo 1: RAG Agent ---")
    demo_rag()

    print("\n--- Demo 2: Tool-Calling Agent ---")
    demo_tool_agent()

    print("\n--- Demo 3: Memory Agent ---")
    demo_memory_agent()

    print("\n✅ All demos complete!")
