import os
import asyncio
from dotenv import load_dotenv

async def test_openai_connection():
    """Test OpenAI API connection with detailed error reporting"""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("🧪 Testing OpenAI API Connection")
    print("=" * 40)
    
    if not api_key:
        print("❌ No API key found!")
        return
    
    print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Test with the stable 0.28.1 API
        import openai
        openai.api_key = api_key
        
        print("📡 Testing API connection...")
        
        # Test with a simple completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello, API test successful!'"}
            ],
            max_tokens=20
        )
        
        result = response['choices'][0]['message']['content']
        print(f"✅ API Test Successful!")
        print(f"📝 Response: {result}")
        
        # Test embeddings
        print("\n🔍 Testing embeddings...")
        embed_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input="test embedding"
        )
        
        embedding = embed_response['data'][0]['embedding']
        print(f"✅ Embedding Test Successful!")
        print(f"📊 Embedding dimension: {len(embedding)}")
        
        return True
        
    except openai.error.AuthenticationError as e:
        print(f"❌ Authentication Error: {str(e)}")
        print("🔧 This means your API key is invalid or incorrectly formatted")
        print("💡 Please check your API key at: https://platform.openai.com/api-keys")
        return False
        
    except openai.error.RateLimitError as e:
        print(f"⚠️ Rate Limit Error: {str(e)}")
        print("💡 Your API key is valid but you've hit rate limits")
        return False
        
    except openai.error.APIError as e:
        print(f"❌ API Error: {str(e)}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_openai_connection())
