"""
Feedback tracking and management system
Logs user feedback on responses and provides analytics
"""

from datetime import datetime
from typing import Optional, Dict, List
from database import (
    UserFeedback,
    UserQuery,
    LLMResponse,
    QueryContext,
    get_db_session,
)
from sqlalchemy import func, and_
import json


def log_query(
    session_id: int,
    user_id: int,
    query_text: str,
    query_type: str = None,
    invoice_uploaded: bool = False,
    invoice_metadata: Dict = None,
) -> int:
    """
    Log a user query
    Returns query_id
    """
    with get_db_session() as db_session:
        query = UserQuery(
            session_id=session_id,
            user_id=user_id,
            query_text=query_text,
            query_type=query_type,
            invoice_uploaded=invoice_uploaded,
            invoice_metadata=invoice_metadata,
        )
        db_session.add(query)
        db_session.commit()
        return query.id


def log_response(
    query_id: int,
    response_text: str,
    response_type: str = "conversational",
    model_name: str = None,
    prompt_template: str = None,
    temperature: float = 0.2,
    total_tokens: int = None,
) -> int:
    """
    Log an LLM response
    Returns response_id
    """
    with get_db_session() as db_session:
        response = LLMResponse(
            query_id=query_id,
            response_text=response_text,
            response_type=response_type,
            model_name=model_name,
            prompt_template=prompt_template,
            temperature=temperature,
            total_tokens=total_tokens,
        )
        db_session.add(response)
        db_session.commit()
        return response.id


def log_retrieved_contexts(
    query_id: int,
    contexts: List[Dict],
) -> List[int]:
    """
    Log retrieved context chunks for a query
    contexts: list of dicts with keys: text, type, level, cluster_id, similarity_score, rank, metadata
    Returns list of context_ids
    """
    context_ids = []
    with get_db_session() as db_session:
        for ctx in contexts:
            context = QueryContext(
                query_id=query_id,
                context_text=ctx.get("text", ""),
                context_type=ctx.get("type", "chunk"),
                level=ctx.get("level"),
                cluster_id=ctx.get("cluster_id"),
                similarity_score=ctx.get("similarity_score"),
                rank=ctx.get("rank"),
                chunk_metadata=ctx.get("metadata"),
            )
            db_session.add(context)
            db_session.commit()
            context_ids.append(context.id)
    return context_ids


def log_feedback(
    query_id: int,
    user_id: int,
    rating: Optional[int] = None,
    is_helpful: Optional[bool] = None,
    comment: Optional[str] = None,
) -> int:
    """
    Log user feedback on a response
    rating: 1-5 stars, or -1 (thumbs down), 1 (thumbs up)
    is_helpful: True/False
    Returns feedback_id
    """
    with get_db_session() as db_session:
        # Check if feedback already exists
        existing_feedback = (
            db_session.query(UserFeedback)
            .filter(UserFeedback.query_id == query_id)
            .first()
        )

        if existing_feedback:
            # Update existing feedback
            if rating is not None:
                existing_feedback.rating = rating
            if is_helpful is not None:
                existing_feedback.is_helpful = is_helpful
            if comment is not None:
                existing_feedback.comment = comment
            existing_feedback.timestamp = datetime.utcnow()
            db_session.commit()
            return existing_feedback.id
        else:
            # Create new feedback
            feedback = UserFeedback(
                query_id=query_id,
                user_id=user_id,
                rating=rating,
                is_helpful=is_helpful,
                comment=comment,
            )
            db_session.add(feedback)
            db_session.commit()
            return feedback.id


def update_query_embedding(query_id: int, embedding: List[float]):
    """Update query with its embedding vector"""
    with get_db_session() as db_session:
        query = db_session.query(UserQuery).filter(UserQuery.id == query_id).first()
        if query:
            query.query_embedding = embedding
            db_session.commit()


def get_feedback_stats(user_id: Optional[int] = None, days: int = 30) -> Dict:
    """
    Get feedback statistics
    Returns dict with average rating, total feedback count, positive/negative counts
    """
    with get_db_session() as db_session:
        query = db_session.query(UserFeedback)

        if user_id:
            query = query.filter(UserFeedback.user_id == user_id)

        # Filter by date range
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(UserFeedback.timestamp >= cutoff_date)

        feedback_list = query.all()

        if not feedback_list:
            return {
                "total_count": 0,
                "average_rating": 0,
                "positive_count": 0,
                "negative_count": 0,
            }

        ratings = [f.rating for f in feedback_list if f.rating is not None]
        helpful_count = sum(1 for f in feedback_list if f.is_helpful is True)
        unhelpful_count = sum(1 for f in feedback_list if f.is_helpful is False)

        return {
            "total_count": len(feedback_list),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "positive_count": helpful_count,
            "negative_count": unhelpful_count,
            "helpful_percentage": (helpful_count / len(feedback_list) * 100)
            if len(feedback_list) > 0
            else 0,
        }


def get_low_quality_queries(threshold: int = 2, limit: int = 20) -> List[Dict]:
    """
    Get queries with low feedback ratings to identify knowledge gaps
    Returns list of dicts with query info and feedback
    """
    with get_db_session() as db_session:
        results = (
            db_session.query(UserQuery, UserFeedback)
            .join(UserFeedback, UserQuery.id == UserFeedback.query_id)
            .filter(UserFeedback.rating <= threshold)
            .order_by(UserFeedback.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "query_id": query.id,
                "query_text": query.query_text,
                "query_type": query.query_type,
                "rating": feedback.rating,
                "comment": feedback.comment,
                "timestamp": feedback.timestamp,
            }
            for query, feedback in results
        ]


def get_high_quality_queries(threshold: int = 4, limit: int = 20) -> List[Dict]:
    """
    Get queries with high feedback ratings to learn from successful patterns
    Returns list of dicts with query info and feedback
    """
    with get_db_session() as db_session:
        results = (
            db_session.query(UserQuery, UserFeedback, LLMResponse)
            .join(UserFeedback, UserQuery.id == UserFeedback.query_id)
            .join(LLMResponse, UserQuery.id == LLMResponse.query_id)
            .filter(UserFeedback.rating >= threshold)
            .order_by(UserFeedback.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "query_id": query.id,
                "query_text": query.query_text,
                "query_type": query.query_type,
                "response_text": response.response_text,
                "rating": feedback.rating,
                "timestamp": feedback.timestamp,
            }
            for query, feedback, response in results
        ]


def get_query_with_contexts(query_id: int) -> Optional[Dict]:
    """
    Get query with its retrieved contexts and response
    """
    with get_db_session() as db_session:
        query = db_session.query(UserQuery).filter(UserQuery.id == query_id).first()

        if not query:
            return None

        contexts = (
            db_session.query(QueryContext)
            .filter(QueryContext.query_id == query_id)
            .order_by(QueryContext.rank)
            .all()
        )

        response = (
            db_session.query(LLMResponse)
            .filter(LLMResponse.query_id == query_id)
            .first()
        )

        feedback = (
            db_session.query(UserFeedback)
            .filter(UserFeedback.query_id == query_id)
            .first()
        )

        return {
            "query": {
                "id": query.id,
                "text": query.query_text,
                "type": query.query_type,
                "timestamp": query.timestamp,
            },
            "contexts": [
                {
                    "text": ctx.context_text,
                    "type": ctx.context_type,
                    "level": ctx.level,
                    "similarity_score": ctx.similarity_score,
                    "rank": ctx.rank,
                }
                for ctx in contexts
            ],
            "response": {
                "text": response.response_text,
                "type": response.response_type,
            }
            if response
            else None,
            "feedback": {
                "rating": feedback.rating,
                "is_helpful": feedback.is_helpful,
                "comment": feedback.comment,
            }
            if feedback
            else None,
        }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing Feedback Tracking System...")

    # Example: Log a query, response, contexts, and feedback
    query_id = log_query(
        session_id=1,
        user_id=1,
        query_text="What is the TDS rate for professional fees?",
        query_type="tds_rate",
    )
    print(f"✅ Query logged with ID: {query_id}")

    response_id = log_response(
        query_id=query_id,
        response_text="Section 194J applies with 10% TDS rate",
        response_type="structured",
        model_name="gpt-4",
    )
    print(f"✅ Response logged with ID: {response_id}")

    contexts = [
        {
            "text": "Section 194J: Professional fees...",
            "type": "chunk",
            "level": 0,
            "cluster_id": 1,
            "similarity_score": 0.92,
            "rank": 1,
        }
    ]
    context_ids = log_retrieved_contexts(query_id, contexts)
    print(f"✅ Contexts logged: {context_ids}")

    feedback_id = log_feedback(query_id=query_id, user_id=1, rating=5, is_helpful=True)
    print(f"✅ Feedback logged with ID: {feedback_id}")

    # Get stats
    stats = get_feedback_stats()
    print(f"✅ Feedback stats: {stats}")
