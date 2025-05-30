"""
Update main.py to use the fallback RAG service
Run this to switch to the fallback implementation
"""

def update_main_py():
    """Update main.py to use fallback RAG service"""
    
    print("🔄 Updating main.py to use fallback RAG service...")
    
    try:
        # Read the current main.py
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Replace the import
        updated_content = content.replace(
            'from enhanced_rag_service import EnhancedRAGService',
            'from enhanced_rag_service_fallback import EnhancedRAGService'
        )
        
        # Write the updated content
        with open('main.py', 'w') as f:
            f.write(updated_content)
        
        print("✅ main.py updated successfully!")
        print("📝 Now using fallback RAG service with Python-based similarity calculation")
        
    except Exception as e:
        print(f"❌ Error updating main.py: {str(e)}")
        raise

if __name__ == "__main__":
    update_main_py()
