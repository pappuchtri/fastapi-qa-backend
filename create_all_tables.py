"""
Database table creation script for Enhanced RAG Q&A System
This script creates all necessary tables for the application including:
- Core Q&A tables (questions, answers, embeddings)
- Document management tables
- User management and authentication
- Human override and review system
- Performance caching and analytics
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging
from document_models import Base
from database import engine

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment variables"""
    # Try different environment variable names
    database_url = (
        os.getenv("DATABASE_URL") or 
        os.getenv("POSTGRES_URL") or 
        os.getenv("POSTGRES_PRISMA_URL") or
        os.getenv("NEON_DATABASE_URL")
    )
    
    if not database_url:
        logger.error("❌ No database URL found in environment variables")
        logger.info("Available environment variables:")
        for key in os.environ:
            if any(db_key in key.upper() for db_key in ['DATABASE', 'POSTGRES', 'NEON']):
                logger.info(f"  {key}: {os.environ[key][:50]}...")
        sys.exit(1)
    
    # Handle SSL requirements for cloud databases
    if "neon.tech" in database_url or "render.com" in database_url:
        if "sslmode" not in database_url:
            database_url += "?sslmode=require"
    
    return database_url

def create_tables():
    """Create all database tables"""
    print("Creating all database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
    """
    database_url = get_database_url()
    logger.info(f"🔗 Connecting to database: {database_url[:50]}...")
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ Connected to PostgreSQL: {version}")
        
        # Create all tables using SQL DDL
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                logger.info("📋 Creating core tables...")
                
                # 1. Users table (for authentication and roles)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'reviewer', 'admin')),
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created users table")
                
                # 2. Questions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS questions (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created questions table")
                
                # 3. Answers table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS answers (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                        text TEXT NOT NULL,
                        confidence_score DECIMAL(3,2) DEFAULT 0.95,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created answers table")
                
                # 4. Embeddings table (using FLOAT array for compatibility)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                        vector FLOAT[] NOT NULL,
                        model_name VARCHAR(100) DEFAULT 'text-embedding-ada-002',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created embeddings table")
                
                # 5. Documents table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        original_filename VARCHAR(255) NOT NULL,
                        file_size INTEGER NOT NULL,
                        content_type VARCHAR(100) DEFAULT 'application/pdf',
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE,
                        processing_status VARCHAR(50) DEFAULT 'uploaded',
                        total_pages INTEGER DEFAULT 0,
                        total_chunks INTEGER DEFAULT 0
                    )
                """))
                logger.info("✅ Created documents table")
                
                # 6. Document chunks table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page_number INTEGER,
                        chunk_embedding FLOAT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created document_chunks table")
                
                # 7. Answer feedback table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS answer_feedback (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                        answer_id INTEGER REFERENCES answers(id) ON DELETE CASCADE,
                        is_helpful BOOLEAN NOT NULL,
                        feedback_text TEXT,
                        confidence_score DECIMAL(3,2),
                        answer_type VARCHAR(50),
                        user_session VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created answer_feedback table")
                
                # 8. Answer overrides table (human corrections)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS answer_overrides (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                        original_answer_id INTEGER REFERENCES answers(id) ON DELETE CASCADE,
                        override_text TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'revoked')),
                        created_by INTEGER REFERENCES users(id),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        approved_at TIMESTAMP
                    )
                """))
                logger.info("✅ Created answer_overrides table")
                
                # 9. Answer reviews table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS answer_reviews (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                        answer_id INTEGER REFERENCES answers(id) ON DELETE CASCADE,
                        override_id INTEGER REFERENCES answer_overrides(id) ON DELETE SET NULL,
                        review_status VARCHAR(20) DEFAULT 'pending' CHECK (review_status IN ('pending', 'approved', 'rejected', 'insufficient')),
                        review_notes TEXT,
                        reviewer_id INTEGER REFERENCES users(id),
                        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_insufficient BOOLEAN DEFAULT FALSE,
                        needs_more_context BOOLEAN DEFAULT FALSE,
                        factual_accuracy_concern BOOLEAN DEFAULT FALSE,
                        compliance_concern BOOLEAN DEFAULT FALSE
                    )
                """))
                logger.info("✅ Created answer_reviews table")
                
                # 10. Performance cache table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_cache (
                        id SERIAL PRIMARY KEY,
                        question_hash VARCHAR(64) UNIQUE NOT NULL,
                        question_text TEXT NOT NULL,
                        cached_answer TEXT NOT NULL,
                        generation_time_ms INTEGER NOT NULL,
                        confidence_score FLOAT NOT NULL,
                        source_type VARCHAR(50) NOT NULL,
                        cache_hits INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """))
                logger.info("✅ Created performance_cache table")
                
                # 11. Chunk usage log table (analytics)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chunk_usage_log (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                        chunk_id INTEGER REFERENCES document_chunks(id) ON DELETE CASCADE,
                        relevance_score FLOAT NOT NULL,
                        position_in_results INTEGER NOT NULL,
                        was_used_in_answer BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created chunk_usage_log table")
                
                # 12. Review tags table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS review_tags (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) UNIQUE NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created review_tags table")
                
                # 13. Review tag associations table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS review_tag_associations (
                        id SERIAL PRIMARY KEY,
                        review_id INTEGER REFERENCES answer_reviews(id) ON DELETE CASCADE,
                        tag_id INTEGER REFERENCES review_tags(id) ON DELETE CASCADE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created review_tag_associations table")
                
                # 14. User sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        session_token VARCHAR(255) UNIQUE NOT NULL,
                        ip_address VARCHAR(50),
                        user_agent VARCHAR(255),
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created user_sessions table")
                
                # 15. Audit log table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        action VARCHAR(100) NOT NULL,
                        entity_type VARCHAR(100) NOT NULL,
                        entity_id INTEGER NOT NULL,
                        details TEXT,
                        ip_address VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info("✅ Created audit_logs table")
                
                logger.info("📊 Creating indexes for performance...")
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_questions_created_at ON questions(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_answers_question_id ON answers(question_id)",
                    "CREATE INDEX IF NOT EXISTS idx_answers_confidence ON answers(confidence_score)",
                    "CREATE INDEX IF NOT EXISTS idx_embeddings_question_id ON embeddings(question_id)",
                    "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)",
                    "CREATE INDEX IF NOT EXISTS idx_document_chunks_page ON document_chunks(page_number)",
                    "CREATE INDEX IF NOT EXISTS idx_feedback_question_id ON answer_feedback(question_id)",
                    "CREATE INDEX IF NOT EXISTS idx_feedback_helpful ON answer_feedback(is_helpful)",
                    "CREATE INDEX IF NOT EXISTS idx_overrides_question_id ON answer_overrides(question_id)",
                    "CREATE INDEX IF NOT EXISTS idx_overrides_status ON answer_overrides(status)",
                    "CREATE INDEX IF NOT EXISTS idx_reviews_status ON answer_reviews(review_status)",
                    "CREATE INDEX IF NOT EXISTS idx_reviews_insufficient ON answer_reviews(is_insufficient)",
                    "CREATE INDEX IF NOT EXISTS idx_cache_hash ON performance_cache(question_hash)",
                    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON performance_cache(expires_at)",
                    "CREATE INDEX IF NOT EXISTS idx_chunk_usage_question ON chunk_usage_log(question_id)",
                    "CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token)",
                    "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at)",
                    "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_logs(entity_type, entity_id)"
                ]
                
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                
                logger.info("✅ Created performance indexes")
                
                logger.info("👤 Creating default admin user...")
                
                # Create default admin user
                conn.execute(text("""
                    INSERT INTO users (username, email, role) 
                    VALUES ('admin', 'admin@example.com', 'admin')
                    ON CONFLICT (username) DO NOTHING
                """))
                
                # Create default reviewer user
                conn.execute(text("""
                    INSERT INTO users (username, email, role) 
                    VALUES ('reviewer', 'reviewer@example.com', 'reviewer')
                    ON CONFLICT (username) DO NOTHING
                """))
                
                logger.info("✅ Created default users")
                
                logger.info("🏷️ Creating default review tags...")
                
                # Create default review tags
                default_tags = [
                    ('low_confidence', 'Answer has low confidence score'),
                    ('gpt_fallback', 'Answer generated by GPT fallback'),
                    ('factual_concern', 'Potential factual accuracy issue'),
                    ('compliance_issue', 'Compliance or regulatory concern'),
                    ('needs_context', 'Answer needs more context'),
                    ('unclear_question', 'Original question was unclear'),
                    ('outdated_info', 'Information may be outdated')
                ]
                
                for tag_name, tag_desc in default_tags:
                    conn.execute(text("""
                        INSERT INTO review_tags (name, description) 
                        VALUES (:name, :description)
                        ON CONFLICT (name) DO NOTHING
                    """), {"name": tag_name, "description": tag_desc})
                
                logger.info("✅ Created default review tags")
                
                # Commit transaction
                trans.commit()
                logger.info("✅ All tables created successfully!")
                
                # Get table count
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                table_count = result.fetchone()[0]
                logger.info(f"📊 Total tables in database: {table_count}")
                
                # Show table sizes
                logger.info("📋 Table summary:")
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        COALESCE(n_tup_ins, 0) as row_count
                    FROM pg_stat_user_tables 
                    ORDER BY tablename
                """))
                
                for row in result:
                    logger.info(f"  {row[1]}: {row[2]} rows")
                
            except Exception as e:
                trans.rollback()
                logger.error(f"❌ Error creating tables: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        raise
    """

def verify_tables():
    """Verify all tables were created correctly"""
    database_url = get_database_url()
    engine = create_engine(database_url)
    
    expected_tables = [
        'users', 'questions', 'answers', 'embeddings', 'documents', 
        'document_chunks', 'answer_feedback', 'answer_overrides', 
        'answer_reviews', 'performance_cache', 'chunk_usage_log',
        'review_tags', 'review_tag_associations', 'user_sessions', 'audit_logs'
    ]
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """))
        
        existing_tables = [row[0] for row in result]
        
        logger.info("🔍 Verifying table creation:")
        for table in expected_tables:
            if table in existing_tables:
                logger.info(f"  ✅ {table}")
            else:
                logger.error(f"  ❌ {table} - MISSING!")
        
        missing_tables = set(expected_tables) - set(existing_tables)
        if missing_tables:
            logger.error(f"❌ Missing tables: {missing_tables}")
            return False
        else:
            logger.info("✅ All tables verified successfully!")
            return True

if __name__ == "__main__":
    logger.info("🚀 Starting database table creation...")
    
    try:
        create_tables()
        #verify_tables()
        logger.info("🎉 Database setup completed successfully!")
        logger.info("🔗 Your FastAPI backend is now ready to use!")
        logger.info("📝 Next steps:")
        logger.info("  1. Start your FastAPI server: python main.py")
        logger.info("  2. Test the API: curl http://localhost:8000/health")
        logger.info("  3. Upload documents and start asking questions!")
        
    except Exception as e:
        logger.error(f"💥 Database setup failed: {str(e)}")
        sys.exit(1)
