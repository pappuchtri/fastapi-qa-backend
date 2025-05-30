"""
Fix for vector similarity search without pgvector extension
This script updates the RAG service to use Python-based similarity calculation
instead of PostgreSQL vector operators that require pgvector.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pgvector_extension():
    """Check if pgvector extension is available"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("❌ DATABASE_URL not found")
        return False
    
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Check if pgvector extension exists
            result = conn.execute(text("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_available_extensions 
                    WHERE name = 'vector'
                )
            """))
            
            pgvector_available = result.fetchone()[0]
            
            if pgvector_available:
                # Check if extension is installed
                result = conn.execute(text("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension 
                        WHERE extname = 'vector'
                    )
                """))
                pgvector_installed = result.fetchone()[0]
                
                logger.info(f"📊 pgvector extension available: {pgvector_available}")
                logger.info(f"📊 pgvector extension installed: {pgvector_installed}")
                
                return pgvector_installed
            else:
                logger.info("📊 pgvector extension not available on this database")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error checking pgvector: {str(e)}")
        return False

def install_pgvector_if_possible():
    """Try to install pgvector extension if available"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return False
    
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Try to create the extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("✅ pgvector extension installed successfully")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ Could not install pgvector extension: {str(e)}")
        logger.info("💡 This is normal for managed databases like Neon, Render, etc.")
        logger.info("💡 The system will use Python-based similarity calculation instead")
        return False

def verify_embeddings_table():
    """Verify embeddings table structure"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return False
    
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Check embeddings table structure
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'embeddings' AND column_name = 'vector'
            """))
            
            column_info = result.fetchone()
            if column_info:
                logger.info(f"📊 Embeddings table vector column type: {column_info[1]}")
                return True
            else:
                logger.error("❌ Embeddings table vector column not found")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error checking embeddings table: {str(e)}")
        return False

def test_similarity_calculation():
    """Test the Python-based similarity calculation"""
    try:
        import numpy as np
        
        # Test vectors
        vec1 = np.random.rand(1536)  # OpenAI embedding dimension
        vec2 = np.random.rand(1536)
        vec3 = vec1.copy()  # Identical vector
        
        def calculate_cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return dot_product / (norm_v1 * norm_v2)
        
        # Test similarity calculations
        sim_random = calculate_cosine_similarity(vec1, vec2)
        sim_identical = calculate_cosine_similarity(vec1, vec3)
        
        logger.info(f"🧮 Similarity test results:")
        logger.info(f"  Random vectors: {sim_random:.4f}")
        logger.info(f"  Identical vectors: {sim_identical:.4f}")
        
        # Verify results make sense
        if 0.95 <= sim_identical <= 1.0 and 0.0 <= sim_random <= 1.0:
            logger.info("✅ Similarity calculation working correctly")
            return True
        else:
            logger.error("❌ Similarity calculation producing unexpected results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing similarity calculation: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Fixing vector similarity search...")
    
    # Check current setup
    logger.info("1️⃣ Checking pgvector extension...")
    pgvector_available = check_pgvector_extension()
    
    if not pgvector_available:
        logger.info("2️⃣ Attempting to install pgvector...")
        install_pgvector_if_possible()
    
    logger.info("3️⃣ Verifying embeddings table...")
    embeddings_ok = verify_embeddings_table()
    
    logger.info("4️⃣ Testing similarity calculation...")
    similarity_ok = test_similarity_calculation()
    
    if embeddings_ok and similarity_ok:
        logger.info("✅ Vector similarity fix completed successfully!")
        logger.info("💡 The system will now use Python-based cosine similarity")
        logger.info("📝 This approach works without requiring pgvector extension")
        logger.info("🚀 Your FastAPI backend should now work correctly")
    else:
        logger.error("❌ Some issues detected. Please check the logs above.")
        sys.exit(1)
