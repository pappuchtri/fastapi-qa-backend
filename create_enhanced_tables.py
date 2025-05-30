from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
import logging

# Import all models to ensure they're registered with the metadata
from database import Base
from override_models import (
    User, AnswerOverride, AnswerReview, PerformanceCache, 
    ChunkUsageLog, ReviewTag, ReviewTagAssociation, UserSession, AuditLog
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all tables defined in the models"""
    load_dotenv()
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        # Create tables
        logger.info("Creating tables...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Tables created successfully")
        
        # Create default admin user
        logger.info("Creating default admin user...")
        from sqlalchemy.orm import sessionmaker
        from override_models import User, UserRole
        
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = Session()
        
        # Check if admin user exists
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                email="admin@example.com",
                role=UserRole.ADMIN
            )
            db.add(admin)
            db.commit()
            logger.info("✅ Default admin user created")
        else:
            logger.info("✅ Admin user already exists")
        
        # Create default reviewer user
        reviewer = db.query(User).filter(User.username == "reviewer").first()
        if not reviewer:
            reviewer = User(
                username="reviewer",
                email="reviewer@example.com",
                role=UserRole.REVIEWER
            )
            db.add(reviewer)
            db.commit()
            logger.info("✅ Default reviewer user created")
        else:
            logger.info("✅ Reviewer user already exists")
        
        # Create default review tags
        from override_models import ReviewTag
        
        default_tags = [
            {"name": "insufficient", "description": "Answer is incomplete or insufficient"},
            {"name": "needs_context", "description": "Answer needs more context from documents"},
            {"name": "factual_error", "description": "Answer contains factual errors"},
            {"name": "compliance_issue", "description": "Answer has compliance or policy issues"},
            {"name": "hallucination", "description": "Answer contains hallucinated information"}
        ]
        
        for tag_data in default_tags:
            tag = db.query(ReviewTag).filter(ReviewTag.name == tag_data["name"]).first()
            if not tag:
                tag = ReviewTag(
                    name=tag_data["name"],
                    description=tag_data["description"]
                )
                db.add(tag)
        
        db.commit()
        logger.info("✅ Default review tags created")
        
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating tables: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_tables()
    if success:
        print("✅ Enhanced tables created successfully")
    else:
        print("❌ Failed to create enhanced tables")
