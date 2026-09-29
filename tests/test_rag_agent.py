import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents import rag_agent


def test_chunking_respects_size_and_overlap():
    doc = Document(page_content="word " * 1000, metadata={"source": "s"})
    chunks = rag_agent.chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 5
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "s" for c in chunks)


def test_load_documents_from_directory(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.md").write_text("ignored")
    docs = rag_agent.load_documents_from_directory(str(tmp_path))
    assert [d.page_content for d in docs] == ["alpha"]


def test_build_chain_requires_input():
    with pytest.raises(ValueError):
        rag_agent.build_rag_chain()


def test_chain_retrieves_and_passes_sources_to_prompt(monkeypatch):
    seen = {}

    class Recorder(FakeListChatModel):
        def _generate(self, messages, *a, **k):
            seen["prompt"] = messages[-1].content
            return super()._generate(messages, *a, **k)

    monkeypatch.setattr(rag_agent, "get_openai_llm", lambda **k: Recorder(responses=["ANSWER"]))
    store = FAISS.from_documents(
        [Document(page_content="RAG retrieves context.", metadata={"source": "doc1"})],
        DeterministicFakeEmbedding(size=32),
    )
    chain = rag_agent.build_rag_chain(vectorstore=store)
    assert chain.invoke("What is RAG?") == "ANSWER"
    assert "[Source: doc1]" in seen["prompt"]
    assert "RAG retrieves context." in seen["prompt"]
    assert "What is RAG?" in seen["prompt"]
