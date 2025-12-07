# RunPod Serverless Deployment Guide - Step by Step

## 📋 Prerequisites

- Docker installerat (lokalt eller på RunPod)
- Docker Hub konto (gratis)
- RunPod konto med Serverless access
- HuggingFace token (för att ladda upp modellen)

---

## 🚀 Step 1: Upload Model to HuggingFace (Recommended)

Detta gör modellen tillgänglig överallt och enklare att använda.

### 1.1 Logga in på HuggingFace

```bash
# På RunPod pod (eller lokalt)
huggingface-cli login --token YOUR_HF_TOKEN
```

### 1.2 Skapa ett nytt repository på HuggingFace

1. Gå till: https://huggingface.co/new
2. Välj "Model"
3. Namn: `your-username/fingpt-v3-float16` (eller valfritt namn)
4. Välj "Private" eller "Public"
5. Klicka "Create repository"

### 1.3 Ladda upp modellen

```bash
# På RunPod pod
cd /runpod-volume/fingpt-v3-float16_202512060944

# Ladda upp till HuggingFace
huggingface-cli upload your-username/fingpt-v3-float16 ./ --repo-type model
```

**Eller om du vill göra det från din lokala dator:**

```bash
# Ladda ner modellen från RunPod volume först (via SSH/scp)
# Sedan lokalt:
huggingface-cli login
huggingface-cli upload your-username/fingpt-v3-float16 ./fingpt-v3-float16_202512060944 --repo-type model
```

---

## 🐳 Step 2: Build Docker Image

### 2.1 Klona/Öppna projektet lokalt

```bash
# Om du inte redan har det lokalt
git clone https://github.com/rickybobby211/FinMin.git
cd FinMin/fingpt/FinGPT_Forecaster/serverless
```

### 2.2 Bygg Docker-imagen

```bash
# Bygg imagen
docker build -t fingpt-forecaster:latest .

# Testa lokalt (valfritt)
docker run -it --rm \
  -e HF_TOKEN="your_token" \
  -e FINNHUB_API_KEY="your_key" \
  -e ADAPTER_PATH="your-username/fingpt-v3-float16" \
  fingpt-forecaster:latest
```

### 2.3 Tagga för Docker Hub

```bash
# Ersätt 'yourusername' med ditt Docker Hub användarnamn
docker tag fingpt-forecaster:latest yourusername/fingpt-forecaster:latest
```

### 2.4 Logga in på Docker Hub

```bash
docker login
# Ange ditt Docker Hub username och password
```

### 2.5 Pusha till Docker Hub

```bash
docker push yourusername/fingpt-forecaster:latest
```

**Detta kan ta 5-10 minuter beroende på din internetanslutning.**

---

## ☁️ Step 3: Create RunPod Serverless Endpoint

### 3.1 Gå till RunPod Dashboard

1. Logga in på: https://www.runpod.io/
2. Gå till **"Serverless"** i menyn
3. Klicka **"New Endpoint"**

### 3.2 Konfigurera Endpoint

**Basic Settings:**
- **Name**: `fingpt-forecaster-v3` (eller valfritt namn)
- **Container Image**: `yourusername/fingpt-forecaster:latest`
- **Container Disk**: `20 GB` (räcker för modellen)

**GPU Settings:**
- **GPU Type**: Välj baserat på behov:
  - `RTX 3090` - Bra balans (24GB VRAM)
  - `RTX 4090` - Snabbare (24GB VRAM)
  - `A100` - Snabbast men dyrare (40GB VRAM)

**Environment Variables:**
Klicka på **"Environment Variables"** och lägg till:

```
HF_TOKEN = hf_your_huggingface_token_here
FINNHUB_API_KEY = your_finnhub_api_key_here
ADAPTER_PATH = your-username/fingpt-v3-float16
```

**Viktigt:** 
- Om du laddade upp till HuggingFace: Använd `your-username/fingpt-v3-float16`
- Om du vill använda officiell FinGPT: Lämna `ADAPTER_PATH` tom eller sätt till `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora`

**Advanced Settings:**
- **Max Workers**: `1` (för att undvika VRAM-problem)
- **Flashboot**: `Enabled` (snabbare cold start)
- **Idle Timeout**: `5 minutes` (eller längre om du vill)

### 3.3 Skapa Endpoint

Klicka **"Create Endpoint"** och vänta på att den skapas (1-2 minuter).

---

## 🧪 Step 4: Test Your API

### 4.1 Hämta API Endpoint URL

Efter att endpointen är skapad:
1. Klicka på din endpoint
2. Kopiera **"Endpoint URL"** (ser ut som: `https://api.runpod.ai/v2/xxxxx`)

### 4.2 Hämta API Key

1. Gå till: https://www.runpod.io/console/user/settings
2. Kopiera din **"API Key"**

### 4.3 Testa med curl

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "ticker": "AAPL",
      "date": "2025-12-06",
      "n_weeks": 3
    }
  }'
```

### 4.4 Testa med Python

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_RUNPOD_API_KEY"
    },
    json={
        "input": {
            "ticker": "AAPL",
            "date": "2025-12-06",
            "n_weeks": 3
        }
    }
)

print(response.json())
```

### 4.5 Testa med Web UI

Om du har `web_ui.py`:

```bash
export RUNPOD_API_URL="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"
export RUNPOD_API_KEY="your_runpod_api_key"
python web_ui.py
```

Öppna `http://localhost:5000` i webbläsaren.

---

## 🔧 Troubleshooting

### Problem: "Model not found"
**Lösning:** Kontrollera att `ADAPTER_PATH` är korrekt i environment variables.

### Problem: "Out of memory"
**Lösning:** 
- Använd större GPU (A100)
- Eller minska `max_new_tokens` i `handler.py`

### Problem: "Cold start timeout"
**Lösning:** 
- Öka timeout i RunPod settings
- Eller använd "Flashboot" för snabbare startup

### Problem: "API returns error"
**Lösning:** 
- Kolla logs i RunPod dashboard
- Verifiera att alla environment variables är satta

---

## 💰 Cost Estimation

**RunPod Serverless Pricing:**
- RTX 3090: ~$0.00029/sekund (~$1.04/timme) när aktiv
- RTX 4090: ~$0.00039/sekund (~$1.40/timme) när aktiv
- A100: ~$0.00069/sekund (~$2.48/timme) när aktiv

**Typisk request:**
- Cold start: ~30-60 sekunder (laddar modellen första gången)
- Warm request: ~10-20 sekunder (modellen redan laddad)

**Kostnad per request:**
- Cold: ~$0.01-0.02
- Warm: ~$0.003-0.006

---

## 📚 Next Steps

1. **Monitor Usage:** Kolla RunPod dashboard för användning och kostnader
2. **Optimize:** Justera `max_new_tokens` för snabbare svar
3. **Scale:** Lägg till fler workers om du har hög trafik
4. **Integrate:** Använd API:et i dina applikationer

---

## ✅ Checklist

- [ ] Modell uppladdad till HuggingFace
- [ ] Docker image byggd och pushad till Docker Hub
- [ ] RunPod Serverless endpoint skapad
- [ ] Environment variables konfigurerade
- [ ] API testat och fungerar
- [ ] Web UI fungerar (valfritt)

**Grattis! Din FinGPT modell är nu live på RunPod Serverless! 🎉**

