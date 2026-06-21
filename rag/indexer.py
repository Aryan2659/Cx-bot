"""
Document Indexer: loads domain documents, chunks, embeds, and saves FAISS indexes.
"""
import os
import pickle
import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import DOCS_DIR, FAISS_INDEX_DIR, EMBED_MODEL, DOMAINS
from pipeline.pii_redactor import redact
from rag.embeddings import initialize, get_langchain_embeddings

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
_initialized = False


def _ensure_initialized():
    global _initialized
    if not _initialized:
        initialize(EMBED_MODEL)
        _initialized = True


def _load_documents(domain: str) -> List[Document]:
    domain_path = os.path.join(DOCS_DIR, domain)
    if not os.path.exists(domain_path):
        logger.warning(f"Document directory not found: {domain_path}")
        return []
    docs = []
    for ext in ["*.txt", "*.md"]:
        try:
            loader = DirectoryLoader(
                domain_path, glob=ext, loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}, silent_errors=True,
            )
            docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"Failed to load {ext} from {domain_path}: {e}")
    for doc in docs:
        doc.page_content = redact(doc.page_content)
        doc.metadata["domain"] = domain
    logger.info(f"Loaded {len(docs)} documents for domain '{domain}'")
    return docs


def build_index(domain: str) -> None:
    _ensure_initialized()
    docs = _load_documents(domain)
    if not docs:
        docs = [Document(
            page_content=f"No documents loaded for {domain}.",
            metadata={"domain": domain, "source": "placeholder"}
        )]
    chunks = _splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks for domain '{domain}'")

    embeddings = get_langchain_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(FAISS_INDEX_DIR, domain)
    vectorstore.save_local(index_path)

    chunks_path = os.path.join(FAISS_INDEX_DIR, f"{domain}_chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Index saved to {index_path}")


def load_index(domain: str) -> FAISS:
    _ensure_initialized()
    index_path = os.path.join(FAISS_INDEX_DIR, domain)
    if not os.path.exists(index_path):
        logger.info(f"Index not found for '{domain}'. Building now...")
        build_index(domain)
    embeddings = get_langchain_embeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def load_chunks(domain: str) -> List[Document]:
    chunks_path = os.path.join(FAISS_INDEX_DIR, f"{domain}_chunks.pkl")
    if not os.path.exists(chunks_path):
        build_index(domain)
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


def build_all_indexes() -> None:
    for domain in DOMAINS:
        logger.info(f"Building index for domain: {domain}")
        build_index(domain)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS + ["all"], default="all")
    args = parser.parse_args()
    if args.domain == "all":
        build_all_indexes()
    else:
        build_index(args.domain)
