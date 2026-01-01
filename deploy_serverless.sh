#!/bin/bash

# Configuration
IMAGE_NAME="rickybobby21/fingpt-forecaster-qwen:v1"
SERVERLESS_DIR="fingpt/FinGPT_Forecaster/serverless"

# Kontrollera om vi är i projektroten
if [ ! -d "fingpt" ]; then
    echo "❌ Fel: Kör detta script från projektets rotmapp (där fingpt-mappen finns)."
    exit 1
fi

echo "========================================================"
echo "🚀 Deploying FinGPT Forecaster to Docker Hub"
echo "========================================================"
echo "Image: $IMAGE_NAME"
echo "Dir:   $SERVERLESS_DIR"
echo "========================================================"

# Navigera till serverless-mappen
cd $SERVERLESS_DIR || { echo "❌ Kunde inte hitta mappen $SERVERLESS_DIR"; exit 1; }

echo "🔨 Building Docker image..."
# --rm tar bort temporära containers för att spara utrymme
# Vi använder --platform linux/amd64 för att säkerställa kompatibilitet med RunPod (om du bygger på M1/M2 Mac t.ex.)
# Då vi redan är på Windows/Linux amd64 är det inte strikt nödvändigt men bra praxis.
docker build --platform linux/amd64 --progress=plain -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Build misslyckades!"
    exit 1
fi

echo "☁️ Pushing to Docker Hub..."
docker push $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "❌ Push misslyckades! Är du inloggad? (docker login)"
    exit 1
fi

echo "🧹 RENSNING AV DISKUTRYMME..."
# 1. Ta bort den gamla imagen som precis blev namnlös (<none>)
docker image prune -f

# 2. Rensa build-cache som är äldre än 24 timmar
# Detta sparar dina 31GB build-cache från att växa okontrollerat
docker builder prune -f --filter "until=24h"

echo "========================================================"
echo "✅ Deployment klar och disken är städad!"
echo "========================================================"
echo "Nästa steg:"
echo "1. Gå till RunPod Console"
echo "2. Redigera din Endpoint"
echo "3. Klicka på Save (för att tvinga fram en omstart och hämta nya imagen)"
echo "========================================================"
