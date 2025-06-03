import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_documents():
    """Debug script to check documents in the database"""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        return
    
    try:
        # Create engine and connect
        engine = create_engine(database_url)
        
        print("🔍 Checking documents table...")
        
        # Check if documents table exists
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'documents'
            """))
            
            if not result.fetchone():
                print("❌ Documents table does not exist!")
                return
            
            print("✅ Documents table exists")
            
            # Check table structure
            print("\n📋 Documents table structure:")
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
            """))
            
            for row in result:
                print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
            
            # Count total documents
            result = conn.execute(text("SELECT COUNT(*) FROM documents"))
            total_docs = result.fetchone()[0]
            print(f"\n📊 Total documents in database: {total_docs}")
            
            if total_docs > 0:
                # Show sample documents
                print("\n📄 Sample documents:")
                result = conn.execute(text("""
                    SELECT id, filename, original_filename, file_size, 
                           content_type, upload_date, processed, processing_status,
                           total_pages, total_chunks
                    FROM documents 
                    ORDER BY upload_date DESC
                    LIMIT 5
                """))
                
                for row in result:
                    print(f"  ID: {row[0]}")
                    print(f"    Filename: {row[1]}")
                    print(f"    Original: {row[2]}")
                    print(f"    Size: {row[3]} bytes")
                    print(f"    Type: {row[4]}")
                    print(f"    Upload Date: {row[5]}")
                    print(f"    Processed: {row[6]}")
                    print(f"    Status: {row[7]}")
                    print(f"    Pages: {row[8]}")
                    print(f"    Chunks: {row[9]}")
                    print("    ---")
            
            # Check document_chunks table
            result = conn.execute(text("SELECT COUNT(*) FROM document_chunks"))
            total_chunks = result.fetchone()[0]
            print(f"\n🧩 Total document chunks: {total_chunks}")
            
    except Exception as e:
        print(f"❌ Error checking documents: {str(e)}")

if __name__ == "__main__":
    debug_documents()
