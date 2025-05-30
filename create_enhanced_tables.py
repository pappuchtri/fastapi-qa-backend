"""
Database migration script to create enhanced RAG tables
Run this script to add override, review, and performance cache tables
"""

from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

def create_enhanced_tables():
    """Create all the new tables for enhanced RAG functionality"""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        return
    
    engine = create_engine(database_url)
    
    # SQL statements to create new tables
    sql_statements = [
        # Users table
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'reviewer', 'admin')),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # Answer overrides table
        """
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
        );
        """,
        
        # Answer reviews table
        """
        CREATE TABLE IF NOT EXISTS answer_reviews (
            id SERIAL PRIMARY KEY,
            question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
            answer_id INTEGER REFERENCES answers(id) ON DELETE CASCADE,
            override_id INTEGER REFERENCES answer_overrides(id) ON DELETE SET NULL,
            review_status VARCHAR(20) DEFAULT 'pending' CHECK (review_status IN ('pending', 'approved', 'rejected', 'insufficient')),
            review_notes TEXT,
            reviewer_id INTEGER REFERENCES users(id),
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_insufficient BOOLEAN DEFAULT false,
            needs_more_context BOOLEAN DEFAULT false,
            factual_accuracy_concern BOOLEAN DEFAULT false,
            compliance_concern BOOLEAN DEFAULT false
        );
        """,
        
        # Performance cache table
        """
        CREATE TABLE IF NOT EXISTS performance_cache (
            id SERIAL PRIMARY KEY,
            question_hash VARCHAR(64) UNIQUE NOT NULL,
            question_text TEXT NOT NULL,
            cached_answer TEXT NOT NULL,
            generation_time_ms INTEGER NOT NULL,
            confidence_score INTEGER NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            cache_hits INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );
        """,
        
        # Create indexes for better performance
        """
        CREATE INDEX IF NOT EXISTS idx_answer_overrides_question_status 
        ON answer_overrides(question_id, status);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_answer_reviews_insufficient 
        ON answer_reviews(is_insufficient, reviewed_at);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_performance_cache_hash 
        ON performance_cache(question_hash);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_performance_cache_expires 
        ON performance_cache(expires_at);
        """,
        
        # Insert default admin user
        """
        INSERT INTO users (username, email, role) 
        VALUES ('admin', 'admin@example.com', 'admin')
        ON CONFLICT (username) DO NOTHING;
        """,
        
        # Insert demo reviewer user
        """
        INSERT INTO users (username, email, role) 
        VALUES ('reviewer', 'reviewer@example.com', 'reviewer')
        ON CONFLICT (username) DO NOTHING;
        """
    ]
    
    try:
        with engine.connect() as conn:
            for i, sql in enumerate(sql_statements, 1):
                print(f"Executing statement {i}/{len(sql_statements)}...")
                conn.execute(text(sql))
                conn.commit()
        
        print("✅ All enhanced RAG tables created successfully!")
        print("\n📋 Created tables:")
        print("- users: User management with roles (user, reviewer, admin)")
        print("- answer_overrides: Human-corrected answers for compliance")
        print("- answer_reviews: Review workflow and quality flags")
        print("- performance_cache: Caching for slow-generating answers")
        print("\n👥 Default users created:")
        print("- admin@example.com (admin role)")
        print("- reviewer@example.com (reviewer role)")
        print("\n🚀 Enhanced RAG system is now ready!")
        
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")

if __name__ == "__main__":
    create_enhanced_tables()
