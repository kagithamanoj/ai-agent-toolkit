"""
Summarize Agent - Map-reduce summarization of long documents.

Long documents overflow an LLM's context window, so the agent first
splits the text into manageable chunks (map), asks the LLM to summarize
each chunk independently, then merges the chunk summaries into one final
summary (reduce).

Uses: LangChain ChatOpenAI for map and reduce steps.

Usage:
    from agents.summarize_agent import SummarizeAgent
    agent = SummarizeAgent()
    result = agent.summarize(long_text)
    print(result["final_summary"])

CLI:
    python -m agents.summarize_agent --file notes.md --chunk-size 3000
    python -m agents.summarize_agent --text "some very long text ..."
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


# ── Text Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 400) -> list[str]:
    """Split text into overlapping chunks, breaking at whitespace.

    Args:
        text: The document to chunk.
        chunk_size: Maximum characters per chunk. Must be positive.
        overlap: Characters carried over from the previous chunk, so ideas
            that span a chunk boundary still get full context. Must be
            smaller than chunk_size.

    Returns:
        A list of chunk strings. Empty/blank text returns an empty list.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be between 0 and chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Back up to the nearest whitespace so words are not split,
        # then re-anchor so each chunk advances by chunk_size - overlap.
        boundary = text.rfind(" ", start, end)
        if boundary <= start:
            boundary = end  # No whitespace found: hard cut.
        chunks.append(text[start:boundary])
        next_start = boundary - overlap
        if next_start <= start:
            next_start = start + 1  # Guarantee forward progress.
        start = next_start

    return chunks


# ── Summarize Agent ─────────────────────────────────────────────────────────────

DEFAULT_MAP_INSTRUCTION = (
    "Summarize the following document chunk in 2-4 sentences. "
    "Capture only the key facts and conclusions; do not add anything "
    "that is not in the chunk."
)

DEFAULT_REDUCE_INSTRUCTION = (
    "Combine the following chunk summaries into one coherent summary of "
    "the whole document. Remove duplication, keep the key facts in "
    "logical order, and write in plain sentences."
)


class SummarizeAgent:
    """Map-reduce summarizer for documents longer than one LLM call can take."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        chunk_size: int = 4000,
        overlap: int = 400,
        map_instruction: str = None,
        reduce_instruction: str = None,
        llm=None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.map_instruction = map_instruction or DEFAULT_MAP_INSTRUCTION
        self.reduce_instruction = reduce_instruction or DEFAULT_REDUCE_INSTRUCTION

        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.map_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "{instruction}"),
                ("human", "{chunk}"),
            ])
            | self.llm
            | StrOutputParser()
        )
        self.reduce_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "{instruction}"),
                ("human", "{summaries}"),
            ])
            | self.llm
            | StrOutputParser()
        )

    def map_chunk(self, chunk: str, index: int = 0) -> str:
        """Summarize a single chunk of the document."""
        print(f"  map [{index + 1}] summarizing {len(chunk)} chars...")
        return self.map_chain.invoke({
            "instruction": self.map_instruction,
            "chunk": chunk,
        })

    def reduce(self, chunk_summaries: list[str]) -> str:
        """Merge chunk summaries into the final summary."""
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        numbered = "\n\n".join(
            f"Chunk {i + 1} summary: {s}"
            for i, s in enumerate(chunk_summaries)
        )
        print(f"  reduce merging {len(chunk_summaries)} chunk summaries...")
        return self.reduce_chain.invoke({
            "instruction": self.reduce_instruction,
            "summaries": numbered,
        })

    def summarize(self, text: str) -> dict:
        """Run the full map-reduce pipeline over text.

        Returns:
            dict with chunk_count, chunk_summaries, final_summary,
            and char counts before/after.
        """
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        if not chunks:
            return {
                "chunk_count": 0,
                "chunk_summaries": [],
                "final_summary": "",
                "input_chars": 0,
                "summary_chars": 0,
            }

        print(f"Split document into {len(chunks)} chunk(s).")
        chunk_summaries = [
            self.map_chunk(chunk, i) for i, chunk in enumerate(chunks)
        ]
        final_summary = self.reduce(chunk_summaries)

        return {
            "chunk_count": len(chunks),
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary,
            "input_chars": len(text),
            "summary_chars": len(final_summary),
        }


# ── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarize a long document with map-reduce."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to summarize.")
    source.add_argument("--file", help="Path to a text file to summarize.")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use.")
    parser.add_argument("--chunk-size", type=int, default=4000,
                        help="Max characters per map chunk.")
    parser.add_argument("--overlap", type=int, default=400,
                        help="Overlap characters between chunks.")
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = Path(args.file).read_text(encoding="utf-8")

    agent = SummarizeAgent(
        model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    result = agent.summarize(text)

    print(f"\nChunks: {result['chunk_count']}, "
          f"input {result['input_chars']} chars -> "
          f"summary {result['summary_chars']} chars")
    print("=" * 60)
    print(result["final_summary"])


if __name__ == "__main__":
    main()
