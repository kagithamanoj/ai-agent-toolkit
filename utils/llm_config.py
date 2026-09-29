"""
AI Agent Toolkit - Utility Configuration
Centralized LLM configuration using environment variables.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Get an OpenAI chat model instance."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_ollama_llm(model: str = "llama3.2", temperature: float = 0.0):
    """Get an Ollama (local) chat model instance."""
    from langchain_community.chat_models import ChatOllama

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def get_embeddings(provider: str = "openai"):
    """Get an embeddings model instance."""
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_vector_store(documents=None, embeddings=None, persist_dir: str = "./vectorstore"):
    """Create or load a FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    if embeddings is None:
        embeddings = get_embeddings()

    if documents:
        return FAISS.from_documents(documents, embeddings)

    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
