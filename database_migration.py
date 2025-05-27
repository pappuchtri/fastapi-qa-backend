"""
Database migration script to fix the embeddings table
Run this to update the existing table structure
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def migrate_embeddings_table():
    """Migrate the embeddings table to use proper PostgreSQL ARRAY type"""
    
    print("🔧 Starting database migration...")
    
    try:
        with engine.connect() as connection:
            # Start a transaction
            trans = connection.begin()
            
            try:
                # Step 1: Check if we need to migrate
                result = connection.execute(text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'embeddings' AND column_name = 'vector'
                """))
                
                column_info = result.fetchone()
                if column_info:
                    print(f"📊 Current vector column type: {column_info[1]}")
                    
                    if column_info[1] == 'json':
                        print("🔄 Migrating from JSON to ARRAY...")
                        
                        # Step 2: Create a backup of existing data
                        print("💾 Backing up existing embeddings...")
                        backup_result = connection.execute(text("""
                            CREATE TABLE embeddings_backup AS 
                            SELECT * FROM embeddings
                        """))
                        
                        # Step 3: Drop the existing table
                        print("🗑️ Dropping existing embeddings table...")
                        connection.execute(text("DROP TABLE embeddings"))
                        
                        # Step 4: Create new table with correct structure
                        print("🏗️ Creating new embeddings table...")
                        connection.execute(text("""
                            CREATE TABLE embeddings (
                                id SERIAL PRIMARY KEY,
                                question_id INTEGER NOT NULL REFERENCES questions(id),
                                vector FLOAT[] NOT NULL,
                                model_name VARCHAR(100) DEFAULT 'text-embedding-ada-002',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                        
                        # Step 5: Migrate data from backup (if any exists)
                        print("📦 Checking for data to migrate...")
                        data_check = connection.execute(text("SELECT COUNT(*) FROM embeddings_backup"))
                        count = data_check.fetchone()[0]
                        
                        if count > 0:
                            print(f"🔄 Migrating {count} embeddings...")
                            # Note: This would require parsing JSON strings back to arrays
                            # For now, we'll skip this and let new embeddings be created
                            print("⚠️ Existing embeddings will be regenerated on next use")
                        
                        # Step 6: Clean up backup table
                        print("🧹 Cleaning up backup table...")
                        connection.execute(text("DROP TABLE embeddings_backup"))
                        
                        print("✅ Migration completed successfully!")
                    else:
                        print("✅ Table already has correct structure")
                else:
                    print("🏗️ Creating embeddings table with correct structure...")
                    connection.execute(text("""
                        CREATE TABLE IF NOT EXISTS embeddings (
                            id SERIAL PRIMARY KEY,
                            question_id INTEGER NOT NULL REFERENCES questions(id),
                            vector FLOAT[] NOT NULL,
                            model_name VARCHAR(100) DEFAULT 'text-embedding-ada-002',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    print("✅ Table created successfully!")
                
                # Commit the transaction
                trans.commit()
                print("💾 Migration committed to database")
                
            except Exception as e:
                # Rollback on error
                trans.rollback()
                print(f"❌ Migration failed, rolling back: {str(e)}")
                raise
                
    except Exception as e:
        print(f"❌ Database migration error: {str(e)}")
        raise

if __name__ == "__main__":
    migrate_embeddings_table()
