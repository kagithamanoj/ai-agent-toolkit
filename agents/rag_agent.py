"""
RAG Agent - Retrieval-Augmented Generation Pattern
Answers questions by retrieving relevant context from a document collection.

Usage:
    python -m agents.rag_agent --query "What is attention in transformers?"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_config import get_openai_llm, get_vector_store

load_dotenv()

# ── RAG Prompt ──────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant. Answer the user's question 
based ONLY on the provided context. If the context doesn't contain enough 
information to answer, say "I don't have enough information to answer that."

Always cite which part of the context you used."""),
    ("human", """Context:
{context}

Question: {question}"""),
])


# ── Document Loading ────────────────────────────────────────────────────────────

def load_documents_from_directory(directory: str, glob: str = "**/*.txt"):
    """Load documents from a local directory."""
    loader = DirectoryLoader(directory, glob=glob, loader_cls=TextLoader)
    return loader.load()


def load_documents_from_urls(urls: list[str]):
    """Load documents from web URLs."""
    loader = WebBaseLoader(urls)
    return loader.load()


def load_documents_from_pdf(pdf_path: str):
    """Load documents from a PDF file."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


# ── Chunking ────────────────────────────────────────────────────────────────────

def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ── RAG Chain ───────────────────────────────────────────────────────────────────

def build_rag_chain(documents=None, vectorstore=None, model: str = "gpt-4o-mini"):
    """
    Build a RAG chain from documents or an existing vector store.

    Args:
        documents: List of Document objects to index (if no vectorstore provided)
        vectorstore: Pre-built vector store (if no documents provided)
        model: LLM model name

    Returns:
        A runnable RAG chain
    """
    llm = get_openai_llm(model=model)

    if vectorstore is None:
        if documents is None:
            raise ValueError("Provide either documents or a vectorstore")
        chunks = chunk_documents(documents)
        vectorstore = get_vector_store(documents=chunks)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Agent - Ask questions about your documents")
    parser.add_argument("--query", "-q", type=str, required=True, help="Question to ask")
    parser.add_argument("--source-dir", "-d", type=str, help="Directory of text files to index")
    parser.add_argument("--source-url", "-u", type=str, nargs="+", help="URLs to index")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="LLM model")
    args = parser.parse_args()

    # Load documents
    documents = []
    if args.source_dir:
        print(f"📂 Loading documents from {args.source_dir}...")
        documents = load_documents_from_directory(args.source_dir)
    elif args.source_url:
        print(f"🌐 Loading documents from {len(args.source_url)} URLs...")
        documents = load_documents_from_urls(args.source_url)
    else:
        # Demo mode with sample text
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content="Transformers use self-attention mechanisms to process sequences in parallel. "
                "The attention mechanism computes weighted sums of values based on query-key similarity. "
                "Multi-head attention allows the model to attend to different representation subspaces.",
                metadata={"source": "demo"},
            ),
            Document(
                page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with "
                "text generation. It first retrieves relevant documents from a knowledge base, then "
                "uses them as context for generating accurate, grounded responses.",
                metadata={"source": "demo"},
            ),
        ]
        print("📝 Using demo documents (pass --source-dir or --source-url for your data)")

    # Build and run
    chain = build_rag_chain(documents=documents, model=args.model)
    print(f"\n🤔 Question: {args.query}")
    print(f"🤖 Answer: {chain.invoke(args.query)}")


if __name__ == "__main__":
    main()
