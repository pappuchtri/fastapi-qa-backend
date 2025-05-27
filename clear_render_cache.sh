#!/bin/bash

echo "🧹 How to Clear Render Build Cache"
echo "================================="

echo "The issue might be that Render is using cached dependencies."
echo "Here's how to force a clean rebuild:"
echo ""

echo "📋 Method 1: Clear Build Cache (Recommended)"
echo "1. Go to your Render dashboard"
echo "2. Click on your service (qa-api-backend)"
echo "3. Go to 'Settings' tab"
echo "4. Scroll down to 'Build & Deploy'"
echo "5. Click 'Clear build cache'"
echo "6. Then click 'Manual Deploy' → 'Deploy latest commit'"
echo ""

echo "📋 Method 2: Force Environment Rebuild"
echo "1. In Render dashboard → Settings → Environment"
echo "2. Add a new environment variable:"
echo "   FORCE_REBUILD=$(date +%s)"
echo "3. Save changes (this will trigger redeploy)"
echo ""

echo "📋 Method 3: Update Build Command (Nuclear Option)"
echo "1. Settings → Build & Deploy"
echo "2. Change Build Command to:"
echo "   pip install --no-cache-dir --force-reinstall -r requirements.txt"
echo "3. Deploy"
echo "4. After successful deploy, change back to:"
echo "   pip install -r requirements.txt"
echo ""

echo "🔍 Check Logs For:"
echo "✅ Installing openai==0.28.1"
echo "✅ OpenAI API key configured successfully (using openai==0.28.1)"
echo "✅ RAG Service initialized with OpenAI 0.28.1 (stable)"
