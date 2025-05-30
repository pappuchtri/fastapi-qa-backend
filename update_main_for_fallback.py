"""
Update main.py to use the fallback RAG service
Run this script to switch to Python-based similarity calculation
"""

def update_main_py():
    """Update main.py to use the fallback RAG service"""
    
    print("🔄 Updating main.py to use fallback RAG service...")
    
    try:
        # Read the current main.py file
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Replace the import statement
        old_import = "from enhanced_rag_service import EnhancedRAGService"
        new_import = "from enhanced_rag_service_fallback import EnhancedRAGService"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✅ Updated import statement")
        else:
            print("⚠️ Original import not found, adding new import")
            # Add the import at the top with other imports
            import_section = "from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status"
            if import_section in content:
                content = content.replace(
                    import_section,
                    f"{import_section}\n{new_import}"
                )
        
        # Write the updated content back to main.py
        with open('main.py', 'w') as f:
            f.write(content)
        
        print("✅ main.py updated successfully!")
        print("🚀 The system now uses Python-based similarity calculation")
        print("📝 No pgvector extension required!")
        
    except Exception as e:
        print(f"❌ Error updating main.py: {str(e)}")
        print("📝 Manual update required:")
        print("   Change: from enhanced_rag_service import EnhancedRAGService")
        print("   To:     from enhanced_rag_service_fallback import EnhancedRAGService")

if __name__ == "__main__":
    update_main_py()
