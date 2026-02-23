# WiLLi — Will's AI Clone with Real-Time RLHF

> A fine-tuned LLM persona trained to represent Will's voice, experience, and personality — with a live reinforcement learning feedback loop that improves the model in real time based on user interactions.

**Live Demo:** [askwilli.dev](https://askwilli.dev)

---

## What is WiLLi?

WiLLi is an AI clone of Will — a conversational agent that answers questions about his career, projects, and technical skills. What makes it different from a standard chatbot is the **live RLHF (Reinforcement Learning from Human Feedback) pipeline** baked in.

Every time Will logs in as admin and rates or corrects a response, that feedback is streamed to a Kafka message broker, consumed by a training loop, and used to fine-tune the model via **DPO (Direct Preference Optimization)** — all while the app is running.

---

## Architecture

```
User → Streamlit UI → FastAPI Backend → Qwen 2.5 3B (LoRA adapter)
                                    ↓
                          Admin correction/feedback
                                    ↓
                        Redpanda (Kafka) message broker
                                    ↓
                        train_loop.py (DPO fine-tuning)
                                    ↓
                        willi_adapter/ (updated LoRA weights)
```

### Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| LLM | Qwen 2.5 3B Instruct (4-bit quantized) |
| Fine-tuning | LoRA + DPO via TRL/PEFT |
| RAG | ChromaDB + LangChain + sentence-transformers |
| Message Broker | Redpanda (Kafka-compatible) via Docker |
| Tunnel | Cloudflare Tunnel → askwilli.dev |
| Training Hardware | Local RTX GPU (CUDA) |

---

## Features

- **Chat with WiLLi** — Ask anything about Will's career, projects, or skills
- **RAG-powered responses** — Answers grounded in Will's actual resume and context documents
- **Live DPO training** — Admin can correct responses and push them directly into training
- **Streaming responses** — Token-by-token output for a natural chat experience
- **Training metrics dashboard** — Real-time view of loss, grad norm, reward margin per training step
- **Architecture + Projects + Resume tabs** — Portfolio embedded directly in the app

---

## Project Structure

```
streaming-rl-llm/
├── app.py                    # Streamlit frontend
├── docker-compose.yml        # Redpanda broker
├── start_willi.bat           # Windows startup script
├── start_redpanda.sh         # WSL startup script
├── rlhf_service/
│   ├── api.py                # FastAPI backend
│   ├── train_loop.py         # DPO training consumer
│   └── ingest.py             # ChromaDB ingestion
├── rag_service/              # RAG retrieval logic
├── stream_service/
│   └── producer.py           # Kafka feedback producer
├── will_context.txt          # Will's background/context
├── resume.pdf                # Resume for RAG + display
├── architecture.html         # Architecture diagram tab
└── projects.html             # Projects showcase tab
```

---

## How the RLHF Loop Works

1. User asks WiLLi a question
2. WiLLi responds using the fine-tuned Qwen model + RAG retrieval
3. Admin logs in and either:
   - Clicks **"Log as Preferred Response"** — marks it as a positive example
   - Types a correction and clicks **"Push to Redpanda for Training"** — creates a DPO pair (chosen vs rejected)
4. The feedback is serialized and published to the `rlhf-feedback` Kafka topic
5. `train_loop.py` consumes the message and runs a DPO training step
6. The LoRA adapter weights are updated and saved to `willi_adapter/`
7. The next response already reflects the correction

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
```

### 3. Create `.streamlit/secrets.toml`
```toml
ADMIN_PASSWORD = "yourpassword"
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Ingest RAG context
```bash
python rag_service/ingest.py
```

### 6. Start everything
Double-click `start_willi.bat` — it will:
- Start Redpanda via WSL/Docker
- Launch FastAPI on port 8000
- Launch Streamlit on port 8501
- Start the Cloudflare tunnel
- Start the training loop

---

## AWS Deployment

See [deployment guide](docs/aws_deploy.md) *(coming soon)* for running on a GPU EC2 instance (g4dn.xlarge recommended).

---

## Secrets & Security

- `.env` and `.streamlit/secrets.toml` are gitignored — never committed
- Redpanda runs locally and is not exposed to the internet
- Admin panel is password-protected
- Cloudflare Tunnel handles SSL and DDoS protection

---

## Why This Project?

This project demonstrates end-to-end ML engineering — not just training a model offline, but building the **full production feedback loop**: data collection, streaming infrastructure, online fine-tuning, and deployment. It's the kind of system that powers real RLHF pipelines at companies like OpenAI and Anthropic, built at portfolio scale.

---

## Author

**Will** — Data Scientist & ML Engineer  
[askwilli.dev](https://askwilli.dev) · [GitHub](https://github.com/wenlong96)
