"""
Learning Engine for RAPTOR-RAG Self-Learning Agent
Analyzes patterns, optimizes retrieval, and improves performance over time
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from database import (
    UserQuery,
    UserFeedback,
    QueryContext,
    LearningParameters,
    get_db_session,
)
from feedback import get_high_quality_queries, get_low_quality_queries
from config import (
    QUERY_SIMILARITY_THRESHOLD,
    MIN_FEEDBACK_COUNT,
    FEEDBACK_THRESHOLD,
    DEFAULT_BFS_THRESHOLD,
    DEFAULT_MAX_DOCS,
)
import json


class LearningEngine:
    """Main learning engine for optimizing retrieval and responses"""

    def __init__(self, embedding_function):
        """
        Args:
            embedding_function: Function that takes text and returns embedding vector
        """
        self.embedding_fn = embedding_function

    def find_similar_queries(
        self,
        query_text: str,
        threshold: float = QUERY_SIMILARITY_THRESHOLD,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Find similar historical queries using semantic similarity
        Returns list of similar queries with their metadata
        """
        query_embedding = self.embedding_fn(query_text)

        with get_db_session() as db_session:
            # Get all queries with embeddings
            historical_queries = (
                db_session.query(UserQuery)
                .filter(UserQuery.query_embedding.isnot(None))
                .all()
            )

            if not historical_queries:
                return []

            # Calculate similarities
            similarities = []
            for hist_query in historical_queries:
                hist_embedding = np.array(hist_query.query_embedding)
                similarity = cosine_similarity(
                    [query_embedding], [hist_embedding]
                )[0][0]

                if similarity >= threshold:
                    similarities.append({
                        "query_id": hist_query.id,
                        "query_text": hist_query.query_text,
                        "similarity": float(similarity),
                        "timestamp": hist_query.timestamp,
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:limit]

    def get_optimal_contexts_for_similar_query(
        self, similar_query_id: int
    ) -> Optional[List[Dict]]:
        """
        Get the contexts that were used for a similar high-rated query
        Returns list of context dicts if query had positive feedback, None otherwise
        """
        with get_db_session() as db_session:
            # Check if query has positive feedback
            feedback = (
                db_session.query(UserFeedback)
                .filter(UserFeedback.query_id == similar_query_id)
                .first()
            )

            if not feedback or (feedback.rating and feedback.rating < FEEDBACK_THRESHOLD):
                return None

            # Get contexts for this query
            contexts = (
                db_session.query(QueryContext)
                .filter(QueryContext.query_id == similar_query_id)
                .order_by(QueryContext.rank)
                .all()
            )

            return [
                {
                    "text": ctx.context_text,
                    "type": ctx.context_type,
                    "level": ctx.level,
                    "cluster_id": ctx.cluster_id,
                    "similarity_score": ctx.similarity_score,
                    "rank": ctx.rank,
                    "metadata": ctx.chunk_metadata,
                }
                for ctx in contexts
            ]

    def optimize_retrieval_parameters(self) -> Dict[str, float]:
        """
        Analyze feedback to optimize BFS threshold and max_docs
        Returns dict with optimized parameters
        """
        with get_db_session() as db_session:
            # Get queries with feedback
            queries_with_feedback = (
                db_session.query(UserQuery, UserFeedback)
                .join(UserFeedback, UserQuery.id == UserFeedback.query_id)
                .filter(UserFeedback.rating.isnot(None))
                .all()
            )

            if len(queries_with_feedback) < MIN_FEEDBACK_COUNT:
                return {
                    "bfs_threshold": DEFAULT_BFS_THRESHOLD,
                    "max_docs": DEFAULT_MAX_DOCS,
                    "confidence": "low",
                }

            # Analyze patterns
            high_rated = []
            low_rated = []

            for query, feedback in queries_with_feedback:
                contexts = (
                    db_session.query(QueryContext)
                    .filter(QueryContext.query_id == query.id)
                    .all()
                )

                context_count = len(contexts)
                avg_similarity = (
                    np.mean([c.similarity_score for c in contexts if c.similarity_score])
                    if contexts
                    else 0
                )

                if feedback.rating >= 4:
                    high_rated.append({
                        "context_count": context_count,
                        "avg_similarity": avg_similarity,
                    })
                elif feedback.rating <= 2:
                    low_rated.append({
                        "context_count": context_count,
                        "avg_similarity": avg_similarity,
                    })

            if not high_rated:
                return {
                    "bfs_threshold": DEFAULT_BFS_THRESHOLD,
                    "max_docs": DEFAULT_MAX_DOCS,
                    "confidence": "low",
                }

            # Calculate optimal parameters based on high-rated queries
            optimal_threshold = np.mean([q["avg_similarity"] for q in high_rated])
            optimal_max_docs = int(np.median([q["context_count"] for q in high_rated]))

            # Adjust based on variance
            threshold_std = np.std([q["avg_similarity"] for q in high_rated])
            adjusted_threshold = max(0.3, optimal_threshold - 0.5 * threshold_std)

            params = {
                "bfs_threshold": float(adjusted_threshold),
                "max_docs": optimal_max_docs,
                "confidence": "high" if len(high_rated) >= 20 else "medium",
                "sample_size": len(queries_with_feedback),
            }

            # Store in database
            self._save_learning_parameter("retrieval_params", params)

            return params

    def identify_knowledge_gaps(self, limit: int = 10) -> List[Dict]:
        """
        Identify topics where the system consistently performs poorly
        Returns list of queries with low ratings grouped by topic
        """
        low_quality = get_low_quality_queries(threshold=2, limit=50)

        if not low_quality:
            return []

        # Group by query type or extract topics
        gaps = {}
        for query_info in low_quality:
            query_type = query_info.get("query_type", "general")
            if query_type not in gaps:
                gaps[query_type] = {
                    "topic": query_type,
                    "count": 0,
                    "average_rating": 0,
                    "examples": [],
                }

            gaps[query_type]["count"] += 1
            gaps[query_type]["average_rating"] += query_info["rating"]
            if len(gaps[query_type]["examples"]) < 3:
                gaps[query_type]["examples"].append(query_info["query_text"])

        # Calculate averages
        for topic in gaps:
            gaps[topic]["average_rating"] /= gaps[topic]["count"]

        # Sort by count
        sorted_gaps = sorted(gaps.values(), key=lambda x: x["count"], reverse=True)
        return sorted_gaps[:limit]

    def suggest_prompt_improvements(self) -> List[str]:
        """
        Analyze high-quality responses to suggest prompt template improvements
        Returns list of prompt suggestions
        """
        high_quality = get_high_quality_queries(threshold=4, limit=30)

        if not high_quality:
            return []

        # Extract common patterns from successful responses
        suggestions = []

        # Analyze response structures
        structured_count = sum(
            1 for q in high_quality if "Section" in q.get("response_text", "")
        )
        if structured_count / len(high_quality) > 0.7:
            suggestions.append(
                "High-rated responses tend to include specific Section references"
            )

        # Analyze query types
        query_types = {}
        for q in high_quality:
            qtype = q.get("query_type", "general")
            query_types[qtype] = query_types.get(qtype, 0) + 1

        if query_types:
            best_type = max(query_types, key=query_types.get)
            suggestions.append(f"Query type '{best_type}' has highest success rate")

        return suggestions

    def get_query_performance_trend(self, days: int = 30) -> Dict:
        """
        Analyze performance trends over time
        Returns dict with metrics showing improvement/decline
        """
        with get_db_session() as db_session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            feedback_over_time = (
                db_session.query(UserFeedback)
                .filter(UserFeedback.timestamp >= cutoff_date)
                .order_by(UserFeedback.timestamp)
                .all()
            )

            if not feedback_over_time:
                return {"trend": "no_data"}

            # Split into first half and second half
            mid_point = len(feedback_over_time) // 2
            first_half = feedback_over_time[:mid_point]
            second_half = feedback_over_time[mid_point:]

            first_avg = np.mean([f.rating for f in first_half if f.rating])
            second_avg = np.mean([f.rating for f in second_half if f.rating])

            return {
                "trend": "improving" if second_avg > first_avg else "declining",
                "first_period_avg": float(first_avg),
                "second_period_avg": float(second_avg),
                "improvement": float(second_avg - first_avg),
                "total_feedback": len(feedback_over_time),
            }

    def _save_learning_parameter(self, param_name: str, param_value: Dict):
        """Save learned parameter to database"""
        with get_db_session() as db_session:
            param = (
                db_session.query(LearningParameters)
                .filter(LearningParameters.parameter_name == param_name)
                .first()
            )

            if param:
                param.parameter_value = param_value
                param.last_updated = datetime.utcnow()
            else:
                param = LearningParameters(
                    parameter_name=param_name, parameter_value=param_value
                )
                db_session.add(param)

            db_session.commit()

    def get_learning_parameter(self, param_name: str) -> Optional[Dict]:
        """Retrieve learned parameter from database"""
        with get_db_session() as db_session:
            param = (
                db_session.query(LearningParameters)
                .filter(LearningParameters.parameter_name == param_name)
                .first()
            )
            return param.parameter_value if param else None


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Testing Learning Engine...")

    embedding_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    engine = LearningEngine(embedding_fn.embed_query)

    # Test optimization
    params = engine.optimize_retrieval_parameters()
    print(f"✅ Optimized parameters: {params}")

    # Test knowledge gaps
    gaps = engine.identify_knowledge_gaps()
    print(f"✅ Knowledge gaps identified: {len(gaps)}")

    # Test trend analysis
    trend = engine.get_query_performance_trend()
    print(f"✅ Performance trend: {trend}")
