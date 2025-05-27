import os
from dotenv import load_dotenv

def debug_openai_key():
    """Debug OpenAI API key configuration"""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("🔍 OpenAI API Key Debug Information")
    print("=" * 50)
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set your OpenAI API key in Render environment variables.")
        return False
    
    print(f"✅ API key found in environment")
    print(f"📏 Key length: {len(api_key)} characters")
    print(f"🔤 Key preview: {api_key[:10]}...{api_key[-4:]}")
    print(f"🎯 Starts with 'sk-': {api_key.startswith('sk-')}")
    print(f"🎯 Starts with 'sk-proj-': {api_key.startswith('sk-proj-')}")
    
    # Check for common issues
    issues = []
    
    if not api_key.startswith('sk-'):
        issues.append("❌ Key doesn't start with 'sk-'")
    
    if len(api_key) < 40:
        issues.append("❌ Key seems too short (should be 51+ characters)")
    
    if ' ' in api_key:
        issues.append("❌ Key contains spaces")
    
    if '\n' in api_key or '\r' in api_key:
        issues.append("❌ Key contains newline characters")
    
    if not api_key.replace('-', '').replace('_', '').isalnum():
        issues.append("❌ Key contains invalid characters")
    
    if issues:
        print("\n🚨 Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ API key format looks correct!")
        return True

if __name__ == "__main__":
    debug_openai_key()
