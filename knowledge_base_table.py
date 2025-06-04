#!/usr/bin/env python3
"""
Create knowledge base table if it doesn't exist
"""

import sys
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from models import KnowledgeBase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_knowledge_base_table():
    """Create knowledge base table and indexes"""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("❌ DATABASE_URL not found in environment variables")
        return False
    
    try:
        engine = create_engine(database_url)
        
        logger.info("🔍 Checking if knowledge_base table exists...")
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'knowledge_base'
            """))
            
            if result.fetchone():
                logger.info("✅ knowledge_base table already exists")
                return True
            
            logger.info("🔨 Creating knowledge_base table...")
            
            # Create the table
            conn.execute(text("""
                CREATE TABLE knowledge_base (
                    id SERIAL PRIMARY KEY,
                    category VARCHAR NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    keywords TEXT[],
                    priority INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR DEFAULT 'system'
                )
            """))
            
            # Create indexes
            conn.execute(text("""
                CREATE INDEX idx_kb_category_active ON knowledge_base(category, is_active)
            """))
            
            conn.execute(text("""
                CREATE INDEX idx_kb_priority ON knowledge_base(priority)
            """))
            
            # Try to create GIN index for keywords (might fail if extension not available)
            try:
                conn.execute(text("""
                    CREATE INDEX idx_kb_keywords ON knowledge_base USING gin(keywords)
                """))
                logger.info("✅ Created GIN index for keywords")
            except Exception as gin_error:
                logger.warning(f"⚠️ Could not create GIN index for keywords: {gin_error}")
                # Create regular index instead
                conn.execute(text("""
                    CREATE INDEX idx_kb_keywords_btree ON knowledge_base(keywords)
                """))
                logger.info("✅ Created B-tree index for keywords instead")
            
            conn.commit()
            
            logger.info("✅ knowledge_base table created successfully!")
            
            # Insert some sample data
            logger.info("📝 Inserting sample knowledge base entries...")
            
            sample_entries = [
                {
                    "category": "AI/ML",
                    "question": "What are vector embeddings?",
                    "answer": "Vector embeddings are numerical representations of text, images, or other data in a high-dimensional space. They capture semantic meaning and allow for similarity comparisons between different pieces of content.",
                    "keywords": ["vector", "embeddings", "AI", "machine learning", "similarity"],
                    "priority": 8
                },
                {
                    "category": "RAG",
                    "question": "How does RAG work?",
                    "answer": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It first searches for relevant documents, then uses that context to generate more accurate and informed responses.",
                    "keywords": ["RAG", "retrieval", "generation", "search", "context"],
                    "priority": 9
                },
                {
                    "category": "Database",
                    "question": "What is PostgreSQL?",
                    "answer": "PostgreSQL is a powerful, open-source relational database management system known for its reliability, feature robustness, and performance. It supports both SQL and JSON querying.",
                    "keywords": ["PostgreSQL", "database", "SQL", "relational", "open source"],
                    "priority": 7
                }
            ]
            
            for entry in sample_entries:
                conn.execute(text("""
                    INSERT INTO knowledge_base (category, question, answer, keywords, priority, created_by)
                    VALUES (:category, :question, :answer, :keywords, :priority, 'system_setup')
                """), {
                    "category": entry["category"],
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "keywords": entry["keywords"],
                    "priority": entry["priority"]
                })
            
            conn.commit()
            logger.info(f"✅ Inserted {len(sample_entries)} sample entries")
            
            # Verify the table
            result = conn.execute(text("SELECT COUNT(*) FROM knowledge_base"))
            count = result.fetchone()[0]
            logger.info(f"✅ knowledge_base table now has {count} entries")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Error creating knowledge_base table: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Setting up knowledge base table...")
    success = create_knowledge_base_table()
    
    if success:
        logger.info("🎉 Knowledge base setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Knowledge base setup failed!")
        sys.exit(1)
