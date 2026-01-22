#!/bin/bash

# =================================================================
# SETUP SCRIPT FÖR RUNPOD TEMPLATE
# Kör detta script EN gång inuti en nystartad RunPod-instans.
# =================================================================

echo "Startar installation..."

# 1. Uppdatera system och installera grundläggande verktyg
apt-get update && apt-get install -y git ninja-build

# 2. Klona repot (om det inte redan finns)
cd /workspace
if [ ! -d "FinMin" ]; then
    git clone https://github.com/rickybobby211/FinMin.git
fi
cd FinMin/fingpt/FinGPT_Forecaster

# 3. Installera dependencies
echo "Installerar Python-bibliotek..."
pip install --upgrade pip
pip install -r training/requirements_training.txt

# 4. Installera Flash Attention (detta tar tid, men görs nu på RunPods snabba datorer)
echo "Installerar Flash Attention (detta kan ta 10-20 minuter)..."
pip install flash-attn --no-build-isolation

# 5. Skapa mappar
mkdir -p /workspace/hf_cache

# 6. Sätt upp miljövariabler permanent i .bashrc
echo 'export HF_HOME="/workspace/hf_cache"' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="/workspace/hf_cache"' >> ~/.bashrc

echo "================================================================="
echo "KLART!"
echo "Nu kan du spara denna pod som en Template:"
echo "1. Gå till RunPod Dashboard -> My Pods"
echo "2. Klicka på menyn (tre punkter) på denna pod."
echo "3. Välj 'Save New Template' (eller 'Commit Container')."
echo "4. Ge den namnet 'fingpt-forecaster-ready'."
echo "================================================================="
