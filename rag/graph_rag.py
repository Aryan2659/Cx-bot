"""
Graph RAG: builds a directed graph from document chunks where edges represent
cross-references. Enables multi-hop reasoning across linked policy documents.
"""
import re
import logging
import pickle
import os
from typing import List, Dict, Set

import networkx as nx
from langchain_core.documents import Document
from rag.indexer import load_chunks
from config import FAISS_INDEX_DIR

logger = logging.getLogger(__name__)

_REF_PATTERNS = [
    re.compile(r"see\s+(?:section|policy|clause|article|paragraph)\s+[\w.]+", re.I),
    re.compile(r"refer\s+to\s+[\w\s]+policy", re.I),
    re.compile(r"as\s+per\s+(?:the\s+)?[\w\s]+policy", re.I),
    re.compile(r"according\s+to\s+(?:the\s+)?[\w\s]+(?:policy|terms|agreement)", re.I),
]

_graph_cache: Dict[str, nx.DiGraph] = {}


def _has_reference(text_a: str, text_b: str) -> bool:
    words_b = set(text_b.lower().split())
    for pattern in _REF_PATTERNS:
        match = pattern.search(text_a)
        if match:
            ref_words = set(match.group().lower().split())
            if ref_words & words_b:
                return True
    return False


def _build_graph(domain: str) -> nx.DiGraph:
    chunks = load_chunks(domain)
    G = nx.DiGraph()
    for i, chunk in enumerate(chunks):
        G.add_node(i, content=chunk.page_content, source=chunk.metadata.get("source", "unknown"))
    for i in range(len(chunks)):
        for j in range(len(chunks)):
            if i != j and _has_reference(chunks[i].page_content, chunks[j].page_content):
                G.add_edge(i, j)
    logger.info(f"Graph RAG: domain='{domain}' nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G


def _get_graph(domain: str) -> nx.DiGraph:
    if domain not in _graph_cache:
        cache_path = os.path.join(FAISS_INDEX_DIR, f"{domain}_graph.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                _graph_cache[domain] = pickle.load(f)
        else:
            G = _build_graph(domain)
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(G, f)
            _graph_cache[domain] = G
    return _graph_cache[domain]


def graph_expand(seed_chunks: List[Document], domain: str, hops: int = 2) -> List[Document]:
    """Expand seed chunks via graph edges for multi-hop reasoning."""
    try:
        G = _get_graph(domain)
    except Exception as e:
        logger.warning(f"Graph RAG unavailable for '{domain}': {e}")
        return seed_chunks

    seed_fps: Set[str] = {chunk.page_content[:80] for chunk in seed_chunks}
    seed_nodes: Set[int] = set()

    for node_id, data in G.nodes(data=True):
        if data.get("content", "")[:80] in seed_fps:
            seed_nodes.add(node_id)

    visited: Set[int] = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(hops):
        next_frontier: Set[int] = set()
        for node in frontier:
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            next_frontier |= neighbors - visited
        visited |= next_frontier
        frontier = next_frontier

    expanded: List[Document] = list(seed_chunks)
    existing_fps = set(seed_fps)

    for node_id in visited - seed_nodes:
        data = G.nodes[node_id]
        content = data.get("content", "")
        fp = content[:80]
        if fp not in existing_fps:
            expanded.append(Document(
                page_content=content,
                metadata={"source": data.get("source", "graph_expanded"), "domain": domain},
            ))
            existing_fps.add(fp)

    logger.debug(f"Graph RAG: {len(seed_chunks)} → {len(expanded)} chunks")
    return expanded
