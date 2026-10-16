"""
Generate test data for testing the learning engine
Creates sample queries, responses, and feedback
"""

from database import get_db_session
from auth import register_user, create_session
from feedback import log_query, log_response, log_retrieved_contexts, log_feedback
from datetime import datetime, timedelta
import random


def create_test_users():
    """Create test users"""
    print("Creating test users...")
    
    users = [
        ("testuser1", "test1@example.com", "password123", "Test User One"),
        ("testuser2", "test2@example.com", "password123", "Test User Two"),
        ("testuser3", "test3@example.com", "password123", "Test User Three"),
    ]
    
    user_ids = []
    for username, email, password, full_name in users:
        user_id = register_user(username, email, password, full_name)
        if user_id:
            print(f"✅ Created user: {username} (ID: {user_id})")
            user_ids.append(user_id)
        else:
            print(f"ℹ️  User {username} already exists")
            # Get existing user ID
            from database import get_user_by_username
            user = get_user_by_username(username)
            if user:
                user_ids.append(user.id)
    
    return user_ids


def generate_sample_queries_and_feedback(user_ids, num_samples=50):
    """Generate sample interactions"""
    print(f"\nGenerating {num_samples} sample queries with feedback...")
    
    sample_queries = [
        ("What is the TDS rate for professional fees?", "tds_rate", 5),
        ("Which section applies to contractor payments?", "section_query", 4),
        ("Explain Section 194J compliance requirements", "general", 5),
        ("TDS on rent payments", "tds_rate", 3),
        ("Section 194C provisions", "section_query", 4),
        ("What is TDS?", "general", 2),
        ("194I applicability on office rent", "section_query", 5),
        ("Professional services TDS deduction", "tds_rate", 4),
        ("Contractor vs professional services TDS", "general", 3),
        ("Section 194H brokerage commission", "section_query", 5),
        ("TDS on software development", "tds_rate", 4),
        ("Payment to non-resident TDS", "section_query", 3),
        ("Lower TDS certificate procedure", "general", 2),
        ("TDS filing deadlines", "general", 4),
        ("Section 194J technical services", "section_query", 5),
    ]
    
    for i in range(num_samples):
        # Random user
        user_id = random.choice(user_ids)
        
        # Create session
        session_id_str = create_session(user_id)
        
        # Get session DB ID
        with get_db_session() as db_session:
            from database import Session
            session_obj = db_session.query(Session).filter(
                Session.session_id == session_id_str
            ).first()
            session_id = session_obj.id if session_obj else 1
        
        # Random query
        query_text, query_type, rating = random.choice(sample_queries)
        
        # Add some variation to queries
        if i > 0:
            query_text = f"{query_text} {random.choice(['', 'please', 'explain', 'details'])}"
        
        # Log query
        query_id = log_query(
            session_id=session_id,
            user_id=user_id,
            query_text=query_text.strip(),
            query_type=query_type,
        )
        
        # Log response
        sample_responses = {
            "tds_rate": "The TDS rate for professional fees under Section 194J is 10%.",
            "section_query": "Section 194J applies to professional or technical services.",
            "general": "TDS (Tax Deducted at Source) is a mechanism to collect tax at source.",
        }
        
        response_text = sample_responses.get(query_type, "This is a sample response.")
        
        log_response(
            query_id=query_id,
            response_text=response_text,
            response_type="conversational",
            model_name="gpt-4",
        )
        
        # Log contexts
        sample_contexts = [
            {
                "text": "Section 194J: TDS on professional or technical services...",
                "type": "chunk",
                "level": 0,
                "cluster_id": 1,
                "similarity_score": 0.85,
                "rank": 1,
            },
            {
                "text": "Professional fees include payments for technical services...",
                "type": "chunk",
                "level": 0,
                "cluster_id": 1,
                "similarity_score": 0.78,
                "rank": 2,
            }
        ]
        
        log_retrieved_contexts(query_id, sample_contexts)
        
        # Log feedback with some randomness
        feedback_rating = rating + random.choice([-1, 0, 0, 1])  # Add noise
        feedback_rating = max(1, min(5, feedback_rating))  # Clamp to 1-5
        
        log_feedback(
            query_id=query_id,
            user_id=user_id,
            rating=feedback_rating,
            is_helpful=feedback_rating >= 3,
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    print(f"✅ Generated {num_samples} sample queries with feedback")


def generate_test_data():
    """Main function to generate all test data"""
    print("\n" + "=" * 80)
    print("GENERATING TEST DATA FOR LEARNING ENGINE")
    print("=" * 80 + "\n")
    
    # Create test users
    user_ids = create_test_users()
    
    if not user_ids:
        print("❌ No users created/found. Exiting.")
        return
    
    # Generate sample data
    generate_sample_queries_and_feedback(user_ids, num_samples=50)
    
    print("\n" + "=" * 80)
    print("✅ TEST DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nYou can now test the learning engine with:")
    print("  python learning_engine.py")
    print("  python analytics.py")
    print("\n")


if __name__ == "__main__":
    generate_test_data()
