"""
Test script to verify database, authentication, and learning engine setup
"""

import sys
from database import init_db, get_db_session, User
from auth import register_user, authenticate_user
from learning_engine import LearningEngine
from analytics import get_query_statistics

def test_database():
    """Test database connection"""
    print("\n" + "=" * 80)
    print("TESTING DATABASE CONNECTION")
    print("=" * 80)
    
    try:
        with get_db_session() as session:
            user_count = session.query(User).count()
            print(f"✅ Database connection successful")
            print(f"   Found {user_count} users in database")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_authentication():
    """Test authentication system"""
    print("\n" + "=" * 80)
    print("TESTING AUTHENTICATION")
    print("=" * 80)
    
    # Try to register a test user
    test_username = f"testuser_{int(__import__('time').time())}"
    test_email = f"{test_username}@test.com"
    test_password = "testpass123"
    
    print(f"\n[1/3] Registering test user: {test_username}")
    user_id = register_user(test_username, test_email, test_password, "Test User")
    
    if user_id:
        print(f"✅ User registered successfully (ID: {user_id})")
    else:
        print(f"❌ User registration failed")
        return False
    
    print(f"\n[2/3] Testing authentication with correct password")
    user = authenticate_user(test_username, test_password)
    if user:
        print(f"✅ Authentication successful")
    else:
        print(f"❌ Authentication failed")
        return False
    
    print(f"\n[3/3] Testing authentication with wrong password")
    user = authenticate_user(test_username, "wrongpassword")
    if not user:
        print(f"✅ Correctly rejected wrong password")
    else:
        print(f"❌ Should have rejected wrong password")
        return False
    
    return True


def test_learning_engine():
    """Test learning engine"""
    print("\n" + "=" * 80)
    print("TESTING LEARNING ENGINE")
    print("=" * 80)
    
    try:
        print("\n[1/2] Initializing learning engine")
        engine = LearningEngine()
        print(f"✅ Learning engine initialized")
        
        print(f"\n[2/2] Testing query similarity")
        similar = engine.find_similar_queries("What is TDS?", top_k=3)
        print(f"✅ Found {len(similar)} similar queries")
        
        if similar:
            for i, (query_id, sim_score, query_text) in enumerate(similar, 1):
                print(f"   {i}. [{sim_score:.2f}] {query_text[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Learning engine test failed: {e}")
        return False


def test_analytics():
    """Test analytics"""
    print("\n" + "=" * 80)
    print("TESTING ANALYTICS")
    print("=" * 80)
    
    try:
        print("\n[1/1] Getting query statistics")
        stats = get_query_statistics()
        print(f"✅ Analytics working")
        print(f"   Total queries: {stats.get('total_queries', 0)}")
        print(f"   Unique users: {stats.get('unique_users', 0)}")
        print(f"   Today's queries: {stats.get('queries_today', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "RAPTOR-RAG SYSTEM VERIFICATION" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    tests = [
        ("Database Connection", test_database),
        ("Authentication", test_authentication),
        ("Learning Engine", test_learning_engine),
        ("Analytics", test_analytics),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<10} {test_name}")
    
    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
