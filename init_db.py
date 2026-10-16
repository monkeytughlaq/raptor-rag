"""
Database initialization script
Creates database tables and admin user
"""

import sys
import os
from database import init_db, get_user_by_username
from auth import create_admin_user
import getpass


def initialize_database():
    """Initialize database tables"""
    print("\n" + "=" * 80)
    print("RAPTOR-RAG DATABASE INITIALIZATION")
    print("=" * 80 + "\n")

    print("[1/2] Creating database tables...")
    try:
        init_db()
        print("✅ Database tables created successfully\n")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        sys.exit(1)

    print("[2/2] Setting up admin user...")
    
    # Check if admin already exists
    existing_admin = get_user_by_username("admin")
    if existing_admin:
        print("ℹ️  Admin user already exists. Skipping admin creation.\n")
        print("=" * 80)
        print("✅ DATABASE INITIALIZATION COMPLETE")
        print("=" * 80 + "\n")
        return

    # Create admin user
    print("\nPlease create an admin user:")
    username = input("Admin username (default: admin): ").strip() or "admin"
    email = input("Admin email: ").strip()
    
    while not email:
        print("❌ Email is required")
        email = input("Admin email: ").strip()
    
    password = getpass.getpass("Admin password: ")
    while not password or len(password) < 6:
        print("❌ Password must be at least 6 characters")
        password = getpass.getpass("Admin password: ")
    
    password_confirm = getpass.getpass("Confirm password: ")
    while password != password_confirm:
        print("❌ Passwords do not match")
        password = getpass.getpass("Admin password: ")
        password_confirm = getpass.getpass("Confirm password: ")
    
    full_name = input("Full name (optional): ").strip() or None

    try:
        admin_id = create_admin_user(username, email, password, full_name)
        if admin_id:
            print(f"\n✅ Admin user '{username}' created successfully (ID: {admin_id})")
        else:
            print(f"\n❌ Failed to create admin user (user might already exist)")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error creating admin user: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ DATABASE INITIALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nYou can now login with:")
    print(f"  Username: {username}")
    print(f"  Password: [the password you entered]")
    print("\n")


if __name__ == "__main__":
    initialize_database()
