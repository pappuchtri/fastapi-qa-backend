import os
import openai
from dotenv import load_dotenv

def check_available_models():
    """Check which OpenAI models are available to your account"""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found!")
        return
    
    openai.api_key = api_key
    
    print("🔍 Checking Available OpenAI Models")
    print("=" * 40)
    
    try:
        # Get list of available models
        models = openai.Model.list()
        
        # Filter for chat models
        chat_models = []
        embedding_models = []
        
        for model in models['data']:
            model_id = model['id']
            if 'gpt' in model_id:
                chat_models.append(model_id)
            elif 'embedding' in model_id:
                embedding_models.append(model_id)
        
        print("💬 Available Chat Models:")
        for model in sorted(chat_models):
            if 'gpt-3.5' in model:
                print(f"  ✅ {model} (Available to all accounts)")
            elif 'gpt-4' in model:
                print(f"  ⚠️  {model} (Requires Tier 1+ access)")
            else:
                print(f"  📝 {model}")
        
        print("\n🔍 Available Embedding Models:")
        for model in sorted(embedding_models):
            print(f"  ✅ {model}")
        
        # Test GPT-3.5-turbo specifically
        print("\n🧪 Testing GPT-3.5-turbo access...")
        test_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello from GPT-3.5-turbo!'"}],
            max_tokens=20
        )
        
        result = test_response['choices'][0]['message']['content']
        print(f"✅ GPT-3.5-turbo test successful: {result}")
        
        # Test GPT-4 access
        print("\n🧪 Testing GPT-4 access...")
        try:
            gpt4_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Say 'Hello from GPT-4!'"}],
                max_tokens=20
            )
            result = gpt4_response['choices'][0]['message']['content']
            print(f"✅ GPT-4 test successful: {result}")
        except Exception as e:
            print(f"❌ GPT-4 not available: {str(e)}")
            print("💡 This is normal for new accounts. Use GPT-3.5-turbo instead.")
        
    except Exception as e:
        print(f"❌ Error checking models: {str(e)}")

if __name__ == "__main__":
    check_available_models()
