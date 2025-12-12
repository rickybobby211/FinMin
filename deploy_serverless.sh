#!/bin/bash

# Configuration
IMAGE_NAME="rickybobby21/fingpt-forecaster:latest"
SERVERLESS_DIR="fingpt/FinGPT_Forecaster/serverless"

# Check if we are in the project root (look for fingpt folder)
if [ ! -d "fingpt" ]; then
    echo "Error: Please run this script from the project root directory (FinGPT/FinGPT)."
    exit 1
fi

echo "========================================================"
echo "🚀 Deploying FinGPT Forecaster to Docker Hub"
echo "========================================================"
echo "Image: $IMAGE_NAME"
echo "Dir:   $SERVERLESS_DIR"
echo "========================================================"

# Navigate to serverless directory
cd $SERVERLESS_DIR

echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "☁️  Pushing to Docker Hub..."
docker push $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "❌ Push failed! Are you logged in? (docker login)"
    exit 1
fi

echo "========================================================"
echo "✅ Deployment Complete!"
echo "========================================================"
echo "Next steps:"
echo "1. Go to RunPod Console"
echo "2. Edit your Endpoint"
echo "3. Click Save (to force a worker restart and pull the new image)"
echo "========================================================"

