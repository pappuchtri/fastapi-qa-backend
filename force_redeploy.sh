#!/bin/bash

echo "🔧 Force Redeployment with OpenAI 0.28.1 Fix"
echo "============================================"

echo "✅ Strategy: Pin to OpenAI 0.28.1 (Stable Version)"
echo "This avoids the v1.0+ migration issues entirely"
echo ""

echo "📋 Changes Made:"
echo "1. requirements.txt: Pinned openai==0.28.1 (stable, well-tested)"
echo "2. rag_service.py: Uses the stable 0.28.1 API format"
echo "3. Removed all v1.0+ API calls that were causing issues"
echo "4. Added better error handling and logging"
echo ""

echo "🔄 Why OpenAI 0.28.1?"
echo "- ✅ Stable and well-tested"
echo "- ✅ No breaking API changes"
echo "- ✅ Works with existing ChatCompletion.create() format"
echo "- ✅ Widely used in production"
echo "- ✅ No migration needed"
echo ""

echo "🚀 Deployment Steps:"
echo "1. Update your GitHub repository with these files"
echo "2. Go to Render dashboard"
echo "3. Settings → Environment → Add this variable:"
echo "   PYTHON_VERSION=3.11"
echo "4. Manual Deploy → Clear build cache → Deploy latest commit"
echo ""

echo "🧪 After deployment, test with:"
echo "curl -X POST https://qa-api-backend.onrender.com/ask \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer dev-api-key-123' \\"
echo "     -d '{\"question\": \"Where is UXCam located?\"}'"
echo ""

echo "✅ Expected: Real GPT-4 answer about UXCam's location!"
