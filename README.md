# 🚀 FastAPI Q&A Backend with RAG

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

A production-ready FastAPI backend with Retrieval-Augmented Generation (RAG) capabilities, designed to work seamlessly with Neon PostgreSQL and OpenAI.

## ✨ Features

- 🤖 **AI-Powered Q&A** with OpenAI GPT-4
- 🔍 **Vector Similarity Search** for intelligent caching
- 🗄️ **Neon PostgreSQL** integration
- 🔐 **API Key Authentication** with rate limiting
- 📊 **Real-time Statistics** and monitoring
- 🎭 **Demo Mode** (works without OpenAI API key)
- 🚀 **One-Click Railway Deployment**

## 🎯 Quick Deploy

### Option 1: One-Click Railway Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

1. Click the "Deploy on Railway" button above
2. Connect your GitHub account
3. Set environment variables (see below)
4. Deploy automatically!

### Option 2: Manual Railway Deploy

1. Fork this repository
2. Sign up at [railway.app](https://railway.app)
3. Create new project from GitHub repo
4. Set environment variables
5. Deploy!

## 🔧 Environment Variables

### Required Variables

\`\`\`bash
DATABASE_URL=postgresql://user:pass@host:5432/db
\`\`\`

### Optional Variables (for full AI functionality)

\`\`\`bash
OPENAI_API_KEY=sk-your-openai-key-here
MASTER_API_KEY=your-secure-master-key
ADDITIONAL_API_KEYS=dev-api-key-123,test-api-key-456
\`\`\`

## 🧪 Testing Your Deployment

### 1. Health Check
\`\`\`bash
curl https://your-app.railway.app/health
\`\`\`

### 2. Test Authentication
\`\`\`bash
curl -H "Authorization: Bearer dev-api-key-123" \
     https://your-app.railway.app/auth/health
\`\`\`

### 3. Ask a Question
\`\`\`bash
curl -X POST https://your-app.railway.app/ask \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev-api-key-123" \
     -d '{"question": "What is artificial intelligence?"}'
\`\`\`

## 📖 API Documentation

Once deployed, visit:
- **API Docs**: `https://your-app.railway.app/docs`
- **ReDoc**: `https://your-app.railway.app/redoc`
- **Health Check**: `https://your-app.railway.app/health`

## 🔑 Development API Keys

These keys are pre-configured for testing:

- `dev-api-key-123`
- `test-api-key-456`
- `demo-key-789`
- `qa-development-key`
- `master-dev-key`

## 🌐 Frontend Integration

After deployment, update your frontend environment variable:

\`\`\`bash
NEXT_PUBLIC_API_URL=https://your-app.railway.app
\`\`\`

## 📊 System Status

The API provides several endpoints to monitor system health:

- `/health` - Basic health check
- `/auth/health` - Authenticated health check
- `/stats` - System statistics (requires auth)
- `/debug/info` - Configuration info

## 🔄 Demo vs Full Mode

### Demo Mode (No OpenAI Key)
- ✅ Full API functionality
- ✅ Database storage
- ✅ Authentication
- 🎭 Simulated AI responses

### Full Mode (With OpenAI Key)
- ✅ Real AI-powered responses
- ✅ Vector similarity search
- ✅ Intelligent caching
- ✅ Embedding storage

## 🛠️ Local Development

\`\`\`bash
# Clone repository
git clone <your-repo>
cd fastapi-backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run application
uvicorn main:app --reload
\`\`\`

## 📞 Support

- 📖 **Documentation**: Check `/docs` endpoint
- 🔍 **Debug Info**: Visit `/debug/info`
- 📊 **Health Status**: Check `/health`
- 🎯 **Test Endpoint**: Try `/test`

## 🚀 Next Steps

1. Deploy using the Railway button above
2. Set your environment variables
3. Test the API endpoints
4. Update your frontend with the new API URL
5. Start asking questions!

Your Q&A system will be fully operational in minutes! 🎉
\`\`\`

```plaintext file=".env.example"
# 🗄️ Database Configuration (Required)
# Get this from your Neon database in Vercel
DATABASE_URL=postgresql://username:password@host:port/database

# 🤖 OpenAI Configuration (Optional - demo mode if not provided)
# Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your_openai_api_key_here

# 🔐 Authentication Configuration
MASTER_API_KEY=your-secure-master-key
ADDITIONAL_API_KEYS=dev-api-key-123,test-api-key-456,demo-key-789,qa-development-key

# ⚡ Rate Limiting Configuration
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# 🧠 RAG Configuration
SIMILARITY_THRESHOLD=0.8
EMBEDDING_MODEL=text-embedding-ada-002
CHAT_MODEL=gpt-4

# 📱 Application Configuration
APP_NAME=Q&A API with RAG and Neon Database
APP_VERSION=3.0.0
DEBUG=False

# 🚀 Railway Configuration (automatically set by Railway)
PORT=8000
RAILWAY_ENVIRONMENT=production
