from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Provide a fallback for development or raise an error
    print("⚠️ WARNING: DATABASE_URL environment variable is not set!")
    DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL debugging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

print("✅ Database configuration loaded:")
print(f"- Database URL configured: {'Yes' if DATABASE_URL else 'No'}")
print("- Connection pooling enabled")
print("- Session management configured")
