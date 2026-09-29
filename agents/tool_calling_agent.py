"""
Tool-Calling Agent - Agent with access to external tools.
Demonstrates how to build an agent that can search the web, do math, and more.

Usage:
    python -m agents.tool_calling_agent --query "What's the weather in Austin, TX?"
"""

import ast
import math
import operator
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


# ── Custom Tools ────────────────────────────────────────────────────────────────

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    k: v for k, v in math.__dict__.items()
    if callable(v) and not k.startswith("_")
}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _safe_eval(node):
    """Evaluate a parsed expression, allowing only numbers, arithmetic and math functions."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python math syntax.

    Args:
        expression: A mathematical expression like '2 + 2' or 'sqrt(144)'
    """
    try:
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information using Tavily.
    
    Args:
        query: The search query string
    """
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query, max_results=3)
        
        output = []
        for r in results.get("results", []):
            output.append(f"**{r['title']}**\n{r['content'][:300]}\nURL: {r['url']}\n")
        
        return "\n---\n".join(output) if output else "No results found."
    except ImportError:
        return (
            "Tavily is not installed. Install with: pip install tavily-python\n"
            "Then set TAVILY_API_KEY in your .env file."
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a local file.
    
    Args:
        filepath: Path to the file to read
    """
    try:
        path = Path(filepath).expanduser()
        if not path.exists():
            return f"File not found: {filepath}"
        if path.stat().st_size > 50_000:
            return f"File too large ({path.stat().st_size} bytes). Max 50KB."
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def current_datetime() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


# ── Agent Builder ───────────────────────────────────────────────────────────────

TOOLS = [calculator, web_search, read_file, current_datetime]

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. 
Use tools when needed to provide accurate, up-to-date information.
Always explain your reasoning and cite your sources when using web search."""


def build_tool_agent(model: str = "gpt-4o-mini", tools: list = None):
    """
    Build a tool-calling agent.

    Args:
        model: LLM model name
        tools: List of tools (defaults to all built-in tools)

    Returns:
        A runnable agent
    """
    if tools is None:
        tools = TOOLS

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools


def run_agent_loop(query: str, model: str = "gpt-4o-mini", max_iterations: int = 5):
    """
    Run an agent loop that processes tool calls iteratively.
    
    Args:
        query: User question
        model: LLM model name
        max_iterations: Maximum number of tool-calling rounds
    """
    from langchain_core.messages import SystemMessage, ToolMessage

    llm_with_tools, tools = build_tool_agent(model=model)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls, we're done
        if not response.tool_calls:
            return response.content

        # Process each tool call
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"  🔧 Calling tool: {tool_name}({tool_args})")

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Max iterations reached. Last response: " + messages[-1].content


# ── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tool-Calling Agent")
    parser.add_argument("--query", "-q", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="LLM model")
    args = parser.parse_args()

    print(f"🤔 Question: {args.query}\n")
    answer = run_agent_loop(args.query, model=args.model)
    print(f"\n🤖 Answer: {answer}")


if __name__ == "__main__":
    main()
