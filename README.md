# WiLLi — Will's AI Clone with Real-Time RLHF

> A living AI clone trained on Will's life, story, and personality — with a real-time reinforcement learning pipeline that improves the model in real time based on direct feedback.

**Live:** [askwilli.dev](https://askwilli.dev)

---

## What is WiLLi?

WiLLi is not a portfolio chatbot. It is a living archive of who Will is — built to let people actually know him, not just read a summary of his credentials.

Chat with a model that speaks as Will. Explore the real-time learning pipeline that powers it. And if it sparks something, build your own.

What makes it different from a standard chatbot is the **live RLHF pipeline** baked in. Every time Will logs in as admin and corrects a response, that feedback streams to a Kafka broker, gets consumed by a training loop, and fine-tunes the model via **DPO (Direct Preference Optimization)** — all while the app is running.

---

## Architecture

```
User → React Frontend (port 3000) → FastAPI Backend (port 8000) → Qwen 2.5 3B (LoRA adapter)
                                                ↓
                                    Admin correction via sidebar
                                                ↓
                                  Redpanda (Kafka) message broker
                                                ↓
                                  train_loop.py (DPO fine-tuning)
                                                ↓
                                  willi_adapter/ (updated LoRA weights)
```

### Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18 (single `index.html`, no build step) |
| Backend API | FastAPI + Uvicorn |
| LLM | Qwen 2.5 3B Instruct (4-bit NF4 quantized) |
| Fine-tuning | LoRA + DPO via TRL / PEFT |
| RAG | ChromaDB + sentence-transformers |
| Message Broker | Redpanda (Kafka-compatible) via Docker/WSL |
| Tunnel | Cloudflare Tunnel → askwilli.dev |
| Training Hardware | Local RTX GPU (CUDA) |

---

## Features

- **Home page** — Intro to WiLLi and Will's story
- **Chat with WiLLi** — Ask anything about Will's life, hobbies, career, or projects
- **RAG-powered responses** — Grounded in Will's actual context documents
- **Live DPO training** — Admin corrects responses and pushes them directly into training
- **Streaming responses** — Token-by-token output via ReadableStream
- **Architecture tab** — Interactive graph of the full pipeline, zoom into each component
- **Projects tab** — Horizontal cinematic scroll through Will's projects
- **Admin sidebar** — Real-time training metrics, logs, and RLHF controls in the chat view

---

## Project Structure

```
streaming-rl-llm/
├── index.html                # React frontend (single file, all tabs)
├── docker-compose.yml        # Redpanda broker config
├── start_willi.bat           # Full startup (with training loop)
├── start_willi_lite.bat      # Lite startup (no training loop)
├── start_redpanda.sh         # WSL Redpanda startup script
├── rlhf_service/
│   ├── api.py                # FastAPI backend
│   ├── train_loop.py         # DPO training consumer
│   └── Dockerfile            # GPU training container
├── rag_service/
│   └── ingest.py             # ChromaDB ingestion
└── stream_service/
    └── producer.py           # Kafka feedback producer
```

---

## How the RLHF Loop Works

1. User asks WiLLi a question
2. WiLLi responds using the fine-tuned Qwen model + RAG retrieval
3. Admin logs in via the admin sidebar and either:
   - Clicks **"Log last response as preferred"** — positive DPO signal
   - Types a correction and clicks **"Push to Redpanda"** — creates a (chosen, rejected) DPO pair
4. Feedback is serialized and published to the `rlhf-feedback` Kafka topic
5. `train_loop.py` consumes the message and runs a DPO gradient step
6. LoRA adapter weights are updated and saved to `willi_adapter/`
7. Next response already reflects the correction

---

## Local Setup (Windows + WSL2)

### Prerequisites
- Windows 10/11 with WSL2
- Docker inside WSL2
- Python 3.10+ with venv
- NVIDIA GPU with CUDA drivers
- Cloudflare account (for tunnel)

### 1. Clone the repo
```bash
git clone https://github.com/wenlong96/streaming-rl-llm.git
cd streaming-rl-llm
```

### 2. Create `.env`
```
KAFKA_BROKER=<your-wsl-ip>:9092
HF_TOKEN=<your-huggingface-token>
TELEGRAM_TOKEN=<optional>
TELEGRAM_CHAT_ID=<optional>
ADMIN_PASSWORD=<your-password>
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add context documents
Place your context file and resume in the project root, then ingest:
```bash
python rag_service/ingest.py
```

### 5. Start everything
Double-click `start_willi.bat` — it will:
- Wait for WSL to initialize
- Start Redpanda via WSL/Docker
- Launch FastAPI on port 8000
- Serve the React frontend on port 3000
- Start the Cloudflare tunnel
- Start the DPO training loop

Or use `start_willi_lite.bat` for a lighter run without the training loop.

---

## Secrets & Security

- `.env` is gitignored — never committed
- Redpanda runs locally and is not exposed to the internet
- Admin panel is password-protected via sessionStorage
- Cloudflare Tunnel handles SSL and DDoS protection

---

## Why This Project?

WiLLi is a personal project built to share Will's life and story — and to inspire others to build their own. There is no better way to express yourself than to make something that thinks and speaks like you.

Technically, it demonstrates end-to-end ML engineering: data collection, streaming infrastructure, online fine-tuning, and deployment. The full production RLHF feedback loop, built at personal project scale.

---

## Author

**Will** — Data Scientist
[askwilli.dev](https://askwilli.dev) · [GitHub](https://github.com/wenlong96)
