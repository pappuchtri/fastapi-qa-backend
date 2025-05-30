"""
Install pgvector extension and update database schema
Run this script to enable vector similarity search
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def install_pgvector():
    """Install pgvector extension and update schema"""
    
    print("🔧 Installing pgvector extension...")
    
    try:
        with engine.connect() as connection:
            # Start a transaction
            trans = connection.begin()
            
            try:
                # Step 1: Install pgvector extension
                print("📦 Installing pgvector extension...")
                try:
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    print("✅ pgvector extension installed successfully")
                except Exception as e:
                    print(f"⚠️ Could not install pgvector extension: {str(e)}")
                    print("📝 Note: You may need to install pgvector manually or use a different similarity method")
                
                # Step 2: Check if embeddings table exists and update it
                print("🔄 Updating embeddings table schema...")
                
                # Check if table exists
                result = connection.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'embeddings'
                    )
                """))
                
                table_exists = result.fetchone()[0]
                
                if table_exists:
                    print("📊 Updating existing embeddings table...")
                    # Backup existing data
                    connection.execute(text("CREATE TABLE IF NOT EXISTS embeddings_backup AS SELECT * FROM embeddings"))
                    
                    # Drop and recreate with vector type
                    connection.execute(text("DROP TABLE embeddings"))
                    
                # Create embeddings table with vector type
                connection.execute(text("""
                    CREATE TABLE embeddings (
                        id SERIAL PRIMARY KEY,
                        question_id INTEGER NOT NULL REFERENCES questions(id),
                        vector vector(1536),
                        model_name VARCHAR(100) DEFAULT 'text-embedding-ada-002',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Step 3: Update document_chunks table if it exists
                print("🔄 Updating document_chunks table schema...")
                
                result = connection.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'document_chunks'
                    )
                """))
                
                chunks_table_exists = result.fetchone()[0]
                
                if chunks_table_exists:
                    # Check if chunk_embedding column exists and update it
                    try:
                        connection.execute(text("ALTER TABLE document_chunks DROP COLUMN IF EXISTS chunk_embedding"))
                        connection.execute(text("ALTER TABLE document_chunks ADD COLUMN chunk_embedding vector(1536)"))
                        print("✅ Updated document_chunks table with vector type")
                    except Exception as e:
                        print(f"⚠️ Could not update document_chunks: {str(e)}")
                
                # Step 4: Create indexes for better performance
                print("📈 Creating vector indexes...")
                try:
                    connection.execute(text("CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100)"))
                    connection.execute(text("CREATE INDEX IF NOT EXISTS document_chunks_vector_idx ON document_chunks USING ivfflat (chunk_embedding vector_cosine_ops) WITH (lists = 100)"))
                    print("✅ Vector indexes created successfully")
                except Exception as e:
                    print(f"⚠️ Could not create vector indexes: {str(e)}")
                    print("📝 Note: Indexes will be created automatically when data is added")
                
                # Commit the transaction
                trans.commit()
                print("💾 Database schema updated successfully!")
                print("\n🎉 pgvector setup complete!")
                print("📝 Note: If pgvector extension installation failed, you may need to:")
                print("   1. Install pgvector on your PostgreSQL server")
                print("   2. Or use a PostgreSQL service that supports pgvector (like Neon, Supabase)")
                
            except Exception as e:
                # Rollback on error
                trans.rollback()
                print(f"❌ Setup failed, rolling back: {str(e)}")
                raise
                
    except Exception as e:
        print(f"❌ Database setup error: {str(e)}")
        raise

if __name__ == "__main__":
    install_pgvector()
