from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Get database URL from environment variables with better error handling
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL environment variable is not set!")
    print("Please set your Neon database URL in Render environment variables.")
    print("Example: postgresql://username:password@host:5432/database")
    raise ValueError("DATABASE_URL environment variable is required")

# Validate DATABASE_URL format
if not DATABASE_URL.startswith(('postgresql://', 'postgres://')):
    print(f"❌ ERROR: Invalid DATABASE_URL format: {DATABASE_URL[:50]}...")
    print("DATABASE_URL should start with 'postgresql://' or 'postgres://'")
    raise ValueError("Invalid DATABASE_URL format")

print(f"✅ Database URL configured: {DATABASE_URL[:30]}...")

try:
    # Create SQLAlchemy engine
    engine = create_engine(DATABASE_URL)
    
    # Test the connection with proper text() wrapper
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        result.fetchone()
    print("✅ Database connection successful!")
    
except Exception as e:
    print(f"❌ Database connection failed: {str(e)}")
    print("Please check your DATABASE_URL and ensure the database is accessible")
    # Don't raise here - let the app start and handle errors gracefully
    print("⚠️ Continuing startup - database errors will be handled at runtime")

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

print("✅ Database setup completed")
