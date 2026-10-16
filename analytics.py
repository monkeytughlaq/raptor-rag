"""
Analytics module for RAPTOR-RAG
Provides insights into system performance and user behavior
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from database import (
    UserQuery,
    UserFeedback,
    Session,
    User,
    QueryContext,
    LLMResponse,
    get_db_session,
)
from sqlalchemy import func, and_, or_
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict


def get_query_frequency_stats(days: int = 30) -> Dict:
    """
    Get query frequency statistics
    Returns dict with total queries, queries per day, peak times
    """
    with get_db_session() as db_session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        queries = (
            db_session.query(UserQuery)
            .filter(UserQuery.timestamp >= cutoff_date)
            .all()
        )

        if not queries:
            return {"total_queries": 0}

        # Group by date
        queries_by_date = defaultdict(int)
        queries_by_hour = defaultdict(int)

        for query in queries:
            date_key = query.timestamp.date()
            hour_key = query.timestamp.hour
            queries_by_date[date_key] += 1
            queries_by_hour[hour_key] += 1

        return {
            "total_queries": len(queries),
            "avg_queries_per_day": len(queries) / days,
            "peak_hour": max(queries_by_hour, key=queries_by_hour.get),
            "queries_by_date": dict(queries_by_date),
            "queries_by_hour": dict(queries_by_hour),
        }


def get_user_engagement_stats() -> Dict:
    """
    Get user engagement statistics
    Returns dict with total users, active users, avg sessions per user
    """
    with get_db_session() as db_session:
        total_users = db_session.query(User).filter(User.is_active == True).count()
        total_sessions = db_session.query(Session).count()

        # Users with at least one query in last 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        active_users = (
            db_session.query(UserQuery.user_id)
            .filter(UserQuery.timestamp >= cutoff_date)
            .distinct()
            .count()
        )

        return {
            "total_users": total_users,
            "active_users_30d": active_users,
            "total_sessions": total_sessions,
            "avg_sessions_per_user": total_sessions / total_users if total_users > 0 else 0,
        }


def get_feedback_quality_stats(days: int = 30) -> Dict:
    """
    Get feedback and quality statistics
    Returns dict with avg rating, feedback rate, quality trends
    """
    with get_db_session() as db_session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Total queries in period
        total_queries = (
            db_session.query(UserQuery)
            .filter(UserQuery.timestamp >= cutoff_date)
            .count()
        )

        # Queries with feedback
        feedback_list = (
            db_session.query(UserFeedback)
            .join(UserQuery, UserFeedback.query_id == UserQuery.id)
            .filter(UserQuery.timestamp >= cutoff_date)
            .all()
        )

        if not feedback_list:
            return {
                "total_queries": total_queries,
                "feedback_count": 0,
                "feedback_rate": 0,
            }

        ratings = [f.rating for f in feedback_list if f.rating is not None]
        helpful_count = sum(1 for f in feedback_list if f.is_helpful is True)

        return {
            "total_queries": total_queries,
            "feedback_count": len(feedback_list),
            "feedback_rate": len(feedback_list) / total_queries if total_queries > 0 else 0,
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "helpful_percentage": helpful_count / len(feedback_list) * 100 if len(feedback_list) > 0 else 0,
            "rating_distribution": {
                "5_stars": sum(1 for r in ratings if r == 5),
                "4_stars": sum(1 for r in ratings if r == 4),
                "3_stars": sum(1 for r in ratings if r == 3),
                "2_stars": sum(1 for r in ratings if r == 2),
                "1_star": sum(1 for r in ratings if r == 1),
            },
        }


def get_query_type_distribution(days: int = 30) -> Dict:
    """
    Get distribution of query types
    Returns dict with counts per query type
    """
    with get_db_session() as db_session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        query_types = (
            db_session.query(UserQuery.query_type, func.count(UserQuery.id))
            .filter(UserQuery.timestamp >= cutoff_date)
            .group_by(UserQuery.query_type)
            .all()
        )

        return {qtype or "general": count for qtype, count in query_types}


def get_retrieval_performance_stats(days: int = 30) -> Dict:
    """
    Get retrieval performance statistics
    Returns dict with avg contexts per query, avg similarity scores
    """
    with get_db_session() as db_session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get contexts for queries in period
        contexts = (
            db_session.query(QueryContext)
            .join(UserQuery, QueryContext.query_id == UserQuery.id)
            .filter(UserQuery.timestamp >= cutoff_date)
            .all()
        )

        if not contexts:
            return {"avg_contexts_per_query": 0}

        # Group by query
        contexts_by_query = defaultdict(list)
        for ctx in contexts:
            contexts_by_query[ctx.query_id].append(ctx)

        avg_contexts = sum(len(ctxs) for ctxs in contexts_by_query.values()) / len(
            contexts_by_query
        )

        similarity_scores = [
            ctx.similarity_score for ctx in contexts if ctx.similarity_score
        ]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

        # Context type distribution
        context_types = defaultdict(int)
        for ctx in contexts:
            context_types[ctx.context_type] += 1

        return {
            "avg_contexts_per_query": avg_contexts,
            "avg_similarity_score": avg_similarity,
            "total_contexts_retrieved": len(contexts),
            "context_type_distribution": dict(context_types),
        }


def get_response_time_stats(days: int = 30) -> Dict:
    """
    Get response time statistics
    Returns dict with avg response time, percentiles
    """
    with get_db_session() as db_session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        queries = (
            db_session.query(UserQuery)
            .filter(
                and_(
                    UserQuery.timestamp >= cutoff_date,
                    UserQuery.processing_time_ms.isnot(None),
                )
            )
            .all()
        )

        if not queries:
            return {"avg_response_time_ms": 0}

        times = [q.processing_time_ms for q in queries]
        import numpy as np

        return {
            "avg_response_time_ms": np.mean(times),
            "median_response_time_ms": np.median(times),
            "p95_response_time_ms": np.percentile(times, 95),
            "p99_response_time_ms": np.percentile(times, 99),
        }


def create_feedback_trend_chart(days: int = 30) -> go.Figure:
    """
    Create a Plotly chart showing feedback trends over time
    """
    with get_db_session() as db_session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        feedback_list = (
            db_session.query(UserFeedback)
            .join(UserQuery, UserFeedback.query_id == UserQuery.id)
            .filter(UserQuery.timestamp >= cutoff_date)
            .order_by(UserQuery.timestamp)
            .all()
        )

        if not feedback_list:
            return go.Figure()

        # Group by date
        feedback_by_date = defaultdict(list)
        for feedback in feedback_list:
            query = (
                db_session.query(UserQuery)
                .filter(UserQuery.id == feedback.query_id)
                .first()
            )
            if query and feedback.rating:
                date_key = query.timestamp.date()
                feedback_by_date[date_key].append(feedback.rating)

        dates = sorted(feedback_by_date.keys())
        avg_ratings = [
            sum(feedback_by_date[date]) / len(feedback_by_date[date]) for date in dates
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=avg_ratings,
                mode="lines+markers",
                name="Average Rating",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Feedback Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Average Rating",
            yaxis=dict(range=[0, 5]),
        )

        return fig


def create_query_type_chart(days: int = 30) -> go.Figure:
    """
    Create a Plotly pie chart showing query type distribution
    """
    distribution = get_query_type_distribution(days)

    if not distribution:
        return go.Figure()

    fig = px.pie(
        values=list(distribution.values()),
        names=list(distribution.keys()),
        title="Query Type Distribution",
    )

    return fig


def generate_analytics_report(days: int = 30) -> Dict:
    """
    Generate comprehensive analytics report
    Returns dict with all key metrics
    """
    return {
        "period_days": days,
        "query_stats": get_query_frequency_stats(days),
        "user_engagement": get_user_engagement_stats(),
        "feedback_quality": get_feedback_quality_stats(days),
        "query_types": get_query_type_distribution(days),
        "retrieval_performance": get_retrieval_performance_stats(days),
        "response_times": get_response_time_stats(days),
        "generated_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing Analytics Module...")

    report = generate_analytics_report(days=30)
    print("\n" + "=" * 80)
    print("ANALYTICS REPORT (Last 30 Days)")
    print("=" * 80)

    for section, data in report.items():
        if section != "generated_at":
            print(f"\n{section.upper().replace('_', ' ')}:")
            for key, value in data.items():
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)
