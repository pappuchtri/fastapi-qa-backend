"""
Check which RAG service is being imported in main.py and fix the import
"""

import os

def check_and_fix_main_imports():
    """Check main.py and fix RAG service imports"""
    
    # Read main.py to see which RAG service is being imported
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        print("🔍 Checking main.py imports...")
        
        # Check for different RAG service imports
        if 'from enhanced_rag_service import EnhancedRAGService' in content:
            print("📦 Found: enhanced_rag_service import")
            return 'enhanced_rag_service'
        elif 'from rag_service import RAGService' in content:
            print("📦 Found: rag_service import")
            return 'rag_service'
        elif 'from enhanced_rag_service_fallback import EnhancedRAGService' in content:
            print("📦 Found: enhanced_rag_service_fallback import")
            return 'enhanced_rag_service_fallback'
        else:
            print("⚠️ No RAG service import found in main.py")
            return None
            
    except FileNotFoundError:
        print("❌ main.py not found")
        return None
    except Exception as e:
        print(f"❌ Error reading main.py: {str(e)}")
        return None

def update_main_to_use_working_rag():
    """Update main.py to use the working RAG service"""
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Replace any enhanced RAG service imports with the basic working one
        content = content.replace(
            'from enhanced_rag_service import EnhancedRAGService',
            'from rag_service import RAGService'
        )
        content = content.replace(
            'from enhanced_rag_service_fallback import EnhancedRAGService',
            'from rag_service import RAGService'
        )
        
        # Replace EnhancedRAGService usage with RAGService
        content = content.replace('EnhancedRAGService()', 'RAGService()')
        
        with open('main.py', 'w') as f:
            f.write(content)
        
        print("✅ Updated main.py to use working RAG service")
        return True
        
    except Exception as e:
        print(f"❌ Error updating main.py: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Checking and fixing RAG service imports...")
    
    current_import = check_and_fix_main_imports()
    
    if current_import and current_import != 'rag_service':
        print(f"🔄 Switching from {current_import} to rag_service...")
        success = update_main_to_use_working_rag()
        
        if success:
            print("✅ Fixed! Now using the working RAG service.")
            print("🚀 Try starting your server again: python main.py")
        else:
            print("❌ Failed to update imports")
    else:
        print("✅ Already using the correct RAG service")
