"""
Enhanced BFS Retrieval with Learning Capabilities
Integrates with learning engine for improved context selection
"""

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import os
from config import DEFAULT_BFS_THRESHOLD, DEFAULT_MAX_DOCS, ENABLE_LEARNING

PERSIST_DIR = "embeddings"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load FAISS vector store
print("[BFS] Loading FAISS vector store...")
try:
    vectorstore = FAISS.load_local(
        PERSIST_DIR, embedding_fn, allow_dangerous_deserialization=True
    )
    print(f"[BFS] ✅ Successfully loaded {vectorstore.index.ntotal} vectors from FAISS")
except Exception as e:
    print(f"[BFS] ❌ Error loading FAISS: {e}")
    vectorstore = None


def retrieve_all_documents():
    if vectorstore is None:
        print("[BFS] ❌ Vector store not loaded")
        return []
    return vectorstore.similarity_search("document content", k=10000)


def cosine_score(query_emb, doc_emb):
    return cosine_similarity([query_emb], [doc_emb])[0][0]


MIN_PARAGRAPH_LENGTH = 100


def bfs_search(
    query: str,
    threshold: float = None,
    max_docs: int = None,
    learning_engine=None,
) -> List:
    """
    Enhanced hierarchical BFS search with learning capabilities
    
    Args:
        query: Search query text
        threshold: Similarity threshold (uses learned value if None)
        max_docs: Maximum documents to return (uses learned value if None)
        learning_engine: Optional LearningEngine instance for optimization
    
    Returns:
        List of LangChain Document objects with contexts
    """
    # Check vectorstore
    if vectorstore is None or vectorstore.index.ntotal == 0:
        print("[BFS] ⚠️ Warning: Vector store is empty or not loaded")
        return []

    # Use learned parameters if available
    if ENABLE_LEARNING and learning_engine:
        try:
            # Check for similar queries
            similar_queries = learning_engine.find_similar_queries(query, threshold=0.85, limit=3)
            
            if similar_queries:
                print(f"[BFS] 🧠 Found {len(similar_queries)} similar queries, checking for reusable contexts...")
                
                # Try to get contexts from most similar high-rated query
                for sim_query in similar_queries:
                    contexts = learning_engine.get_optimal_contexts_for_similar_query(
                        sim_query["query_id"]
                    )
                    if contexts:
                        print(f"[BFS] ✅ Reusing contexts from similar query (similarity: {sim_query['similarity']:.2f})")
                        # Convert stored contexts back to Document objects
                        from langchain_core.documents import Document
                        return [
                            Document(
                                page_content=ctx["text"],
                                metadata={
                                    "type": ctx["type"],
                                    "level": ctx["level"],
                                    "cluster": ctx.get("cluster_id"),
                                    "chunk_type": ctx["type"],
                                }
                            )
                            for ctx in contexts[:max_docs] if max_docs else contexts
                        ]
            
            # Get optimized retrieval parameters
            learned_params = learning_engine.get_learning_parameter("retrieval_params")
            if learned_params and learned_params.get("confidence") in ["high", "medium"]:
                if threshold is None:
                    threshold = learned_params.get("bfs_threshold", DEFAULT_BFS_THRESHOLD)
                    print(f"[BFS] 🧠 Using learned threshold: {threshold:.3f}")
                if max_docs is None:
                    max_docs = learned_params.get("max_docs", DEFAULT_MAX_DOCS)
                    print(f"[BFS] 🧠 Using learned max_docs: {max_docs}")
        except Exception as e:
            print(f"[BFS] ⚠️ Learning integration error: {e}, falling back to defaults")

    # Fall back to defaults if not set
    if threshold is None:
        threshold = DEFAULT_BFS_THRESHOLD
    if max_docs is None:
        max_docs = DEFAULT_MAX_DOCS

    query_emb = embedding_fn.embed_query(query)
    docs = retrieve_all_documents()

    if not docs:
        print("[BFS] ⚠️ Warning: No documents found in vector store")
        return []

    # Organize nodes by level
    levels = {}
    for doc in docs:
        level = doc.metadata.get("level", -1)
        levels.setdefault(level, []).append(doc)

    if not levels:
        print("[BFS] ⚠️ Warning: No valid levels found in documents")
        return []

    max_level = max(levels.keys())

    context_nodes = []
    queue = [(max_level, doc) for doc in levels[max_level]]

    visited = set()
    while queue and len(context_nodes) < max_docs:
        next_queue = []
        for level, doc in queue:
            doc_id = (
                level,
                doc.metadata.get("cluster", doc.metadata.get("chunk_index", -1)),
            )
            if doc_id in visited:
                continue
            visited.add(doc_id)

            doc_emb = embedding_fn.embed_query(doc.page_content)
            score = cosine_score(query_emb, doc_emb)

            if score >= threshold:
                context_nodes.append(doc)
                if level > 0:
                    children = levels.get(level - 1, [])
                    next_queue.extend(
                        [
                            (level - 1, child)
                            for child in children
                            if child.metadata.get("cluster")
                            == doc.metadata.get("cluster")
                        ]
                    )
        queue = next_queue

    # Append raw chunks (level 0)
    chunks = [doc for doc in levels.get(0, []) if doc not in context_nodes]
    chunks_sorted = sorted(
        chunks,
        key=lambda d: cosine_score(query_emb, embedding_fn.embed_query(d.page_content)),
        reverse=True,
    )
    context_nodes.extend(chunks_sorted[: max(1, max_docs - len(context_nodes))])

    # Filter paragraph chunks by minimum length
    filtered_context_nodes = []
    for doc in context_nodes:
        if doc.metadata.get("chunk_type") == "paragraph":
            if len(doc.page_content) >= MIN_PARAGRAPH_LENGTH:
                filtered_context_nodes.append(doc)
        else:
            filtered_context_nodes.append(doc)
    context_nodes = filtered_context_nodes

    # Always include the most relevant table_cluster chunk
    table_clusters = [
        doc for doc in docs if doc.metadata.get("chunk_type") == "table_cluster"
    ]
    if table_clusters:
        best_table = max(
            table_clusters,
            key=lambda d: cosine_score(query_emb, embedding_fn.embed_query(d.page_content)),
        )
        if best_table not in context_nodes:
            context_nodes.append(best_table)

    print(f"[BFS] 📊 Retrieved {len(context_nodes)} contexts (threshold: {threshold:.3f}, max: {max_docs})")
    return context_nodes


def format_contexts_for_logging(contexts: List) -> List[Dict]:
    """
    Convert LangChain Document objects to dict format for database logging
    """
    formatted = []
    for rank, doc in enumerate(contexts, start=1):
        formatted.append({
            "text": doc.page_content,
            "type": doc.metadata.get("type", "chunk"),
            "level": doc.metadata.get("level"),
            "cluster_id": doc.metadata.get("cluster"),
            "similarity_score": doc.metadata.get("similarity_score"),
            "rank": rank,
            "metadata": doc.metadata,
        })
    return formatted


if __name__ == "__main__":
    if vectorstore is None:
        print("[BFS] Cannot run test - vector store not loaded")
        exit(1)

    # Test query for TDS
    query = "What are the TDS provisions for professional services?"

    print(f"\n{'=' * 80}")
    print(f"🔍 QUERY: {query}")
    print(f"{'=' * 80}")

    # Use lower threshold to get more results
    results = bfs_search(query, threshold=0.2, max_docs=15)

    print(f"\n📊 FOUND {len(results)} RELEVANT DOCUMENTS:")
    print(f"📚 Total vectors in store: {vectorstore.index.ntotal}")
    print("-" * 80)

    for i, doc in enumerate(results[:5], 1):  # Show first 5
        metadata = doc.metadata
        level = metadata.get("level", "N/A")
        doc_type = metadata.get("type", "unknown")

        print(f"\n📄 DOCUMENT #{i}")
        print(f"📊 Level: {level} | 🏷️ Type: {doc_type}")
        print(f"📏 Length: {len(doc.page_content)} characters")
        print(f"\n📄 CONTENT (first 200 chars):")
        print("-" * 40)
        print(doc.page_content[:200] + "...")
        print("-" * 40)
