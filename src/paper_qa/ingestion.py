"""RAG pipeline for paper ingestion using LangChain."""

from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings


class PaperStore:
    """In-memory vector store for material science papers."""

    def __init__(self, embedding_fn: Embeddings):
        self.embeddings = embedding_fn
        self.vectorstore: Optional[FAISS] = None
        self.documents: list[Document] = []

    def ingest_text(self, text: str, metadata: dict | None = None) -> int:
        """Ingest a single text blob (e.g. from a paper)."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
        self.documents.extend(docs)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        return len(chunks)

    def ingest_paper_file(self, filepath: str | Path) -> int:
        """Load a .txt file from the papers directory."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Paper not found: {path}")
        loader = TextLoader(str(path))
        docs = loader.load()
        metadata = {"source": path.name, "path": str(path.absolute())}
        return self.ingest_text(docs[0].page_content, metadata=metadata)

    def ingest_all_papers(self) -> dict[str, int]:
        """Ingest all .txt files from the configured papers directory."""
        papers_dir = Path(settings.papers_dir)
        if not papers_dir.exists():
            return {}
        results = {}
        for f in sorted(papers_dir.glob("*.txt")):
            try:
                count = self.ingest_paper_file(f)
                results[f.name] = count
            except Exception as e:
                results[f.name] = 0
        return results

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Retrieve the top-k most relevant document chunks."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    @property
    def is_loaded(self) -> bool:
        return self.vectorstore is not None
