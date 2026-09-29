"""
Memory Agent - Conversational agent with persistent memory.
Demonstrates conversation history, summary memory, and entity tracking.

Usage:
    python -m agents.memory_agent
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


# ── Memory Store ────────────────────────────────────────────────────────────────

class ConversationMemory:
    """Simple conversation memory with optional summarization."""

    def __init__(self, max_messages: int = 20):
        self.messages: list = []
        self.max_messages = max_messages
        self.summary: str = ""

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
        self._trim()

    def get_messages(self) -> list:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"Previous conversation summary: {self.summary}"))
        result.extend(self.messages)
        return result

    def _trim(self):
        if len(self.messages) > self.max_messages:
            # Keep the most recent messages, summarize old ones
            old = self.messages[: len(self.messages) - self.max_messages]
            self.messages = self.messages[-self.max_messages :]
            old_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in old
            )
            self.summary += f"\n{old_text}"

    def clear(self):
        self.messages = []
        self.summary = ""

    @property
    def message_count(self) -> int:
        return len(self.messages)


# ── Memory Agent ────────────────────────────────────────────────────────────────

class MemoryAgent:
    """Conversational agent with memory."""

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = None):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.memory = ConversationMemory()
        self.system_prompt = system_prompt or (
            "You are a friendly, helpful AI assistant. You remember the full "
            "conversation history and refer back to earlier topics when relevant. "
            "Be concise but thorough."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def chat(self, user_input: str) -> str:
        """Send a message and get a response with memory."""
        history = self.memory.get_messages()

        response = self.chain.invoke({
            "history": history,
            "input": user_input,
        })

        # Save to memory
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(response)

        return response

    def reset(self):
        """Clear conversation memory."""
        self.memory.clear()

    @property
    def history_length(self) -> int:
        return self.memory.message_count


# ── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    print("💬 Memory Agent — Interactive Chat")
    print("    Type 'quit' to exit, 'reset' to clear memory, 'history' to see message count\n")

    agent = MemoryAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("👋 Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 Memory cleared.\n")
            continue
        if user_input.lower() == "history":
            print(f"📊 Messages in memory: {agent.history_length}\n")
            continue

        response = agent.chat(user_input)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    main()
