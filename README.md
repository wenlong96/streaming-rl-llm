# WiLLi — Will's AI Clone with Real-Time RLHF

A living AI clone trained on Will's life, story, and personality, with a real-time reinforcement learning pipeline that improves the model based on direct preference feedback.

**Live:** [askwilli.dev](https://askwilli.dev)

---

## What is WiLLi?

WiLLi is a living archive of who Will is. Chat with a model that speaks as Will, explore the live training pipeline that powers it, and play chess against a clone tuned on Will's actual chess.com games.

What makes WiLLi different from a standard chatbot is that **the feedback loop is closed and visible**. Every visitor thumbs-up or thumbs-down is captured as a preference signal. Curated signals stream into a Kafka-protocol broker, get consumed by a continuously running training loop, and update the model's LoRA adapter via DPO (Direct Preference Optimization) — while the app stays live. New visitors talk to a model that already reflects yesterday's feedback.

---

## Architecture

```
                       ┌─────────────────────────────────┐
                       │   React Frontend (port 3000)    │
                       │   Chat · Architecture · Chess   │
                       └──────────────┬──────────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────────┐
                       │   FastAPI Backend (port 8000)   │
                       │   Qwen 2.5 3B + LoRA adapter    │
                       │   ChromaDB RAG retrieval        │
                       └──────────────┬──────────────────┘
                                      │
                                      ▼
                  visitor 👍/👎 → /feedback endpoint
                                      │
                                      ▼
                       ┌─────────────────────────────────┐
                       │  Redpanda (Kafka-API broker)    │
                       │  rlhf-pending     (audit topic) │
                       │  rlhf-feedback    (curated)     │
                       └──────────────┬──────────────────┘
                                      │
                              admin curates pending
                                      │
                                      ▼
                       ┌─────────────────────────────────┐
                       │   train_loop.py                 │
                       │   TRL DPOTrainer (β=0.1)        │
                       │   Online streaming gradient     │
                       └──────────────┬──────────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────────┐
                       │   willi_adapter_checkpoints/    │
                       │   v124, v125, v126…             │
                       │   hot-reload into live model    │
                       └─────────────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| Frontend | React 18 (single `index.html`, no build step) |
| Backend API | FastAPI + Uvicorn, streaming via `transformers.TextIteratorStreamer` |
| LLM | Qwen 2.5 3B Instruct, 4-bit NF4 quantized |
| Fine-tuning | LoRA (r=16) + DPO via TRL / PEFT |
| RAG | ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`) |
| Message broker | Redpanda (Kafka-protocol compatible) via Docker/WSL |
| Chess engine | Stockfish 10.0.2 (Web Worker) + chess.js |
| Tunnel | Cloudflare Tunnel → askwilli.dev |
| Training hardware | Local RTX GPU (CUDA) |

---

## Features

- **Home** — Intro to WiLLi and Will's story
- **Chat with WiLLi** — Conversational AI clone, streaming token-by-token, RAG-grounded in Will's actual context documents
- **Architecture** — Interactive graph of the full pipeline; click components to drill in
- **Projects** — Cinematic horizontal scroll through Will's projects
- **Playground (Chess)** — Play chess against six different opponents, including a "Will" mode that uses Will's real chess.com opening repertoire and reacts to moves with shitposter commentary
- **Admin panel** — Real-time training metrics (loss + reward margin sparklines), pending feedback queue with promote/dismiss controls, adapter checkpoint history with one-click rollback

---

## How the RLHF Loop Works

The loop is deliberately split into capture, curation, and training stages:

### 1. Capture
A visitor chats with WiLLi and clicks 👍 or 👎. The frontend POSTs to `/feedback` with the question, the response, the rating, and conversation context. The API:
- Appends the event to `pending_feedback.jsonl` (durable file log)
- Publishes to Redpanda topic `rlhf-pending` (audit topic — captures everything)

### 2. Curation
The admin panel surfaces pending feedback events. For each one, the admin can:
- **Promote** a 👍 event — the response becomes a "preferred" exemplar
- **Promote** a 👎 event — the admin writes a corrected response, and the (corrected, original) pair becomes a preference pair
- **Dismiss** an event — drops it (filters trolls, low-quality questions, misclicks)

Promoted pairs publish to Redpanda topic `rlhf-feedback`. The two-topic split (everything vs. curated) means the trainer never sees unfiltered noise, and you keep a complete record of what visitors actually clicked.

### 3. Training
`train_loop.py` runs continuously, polling `rlhf-feedback`. For each new preference pair:
- Loads the pair into a TRL `DPOTrainer`
- Runs **one DPO gradient step** with `β=0.1`
- Saves a new adapter checkpoint to `willi_adapter_checkpoints/v{N}/`
- Updates `training_history.jsonl` with loss + reward margin
- Marks the new version as `active` in the manifest

The Kafka consumer uses manual offset commits — if a step fails (OOM, model crash), the offset doesn't commit, and on restart the trainer replays from the last successful step. Feedback events are durable across trainer crashes.

### 4. Hot-reload
The API hot-reloads the new adapter via `load_active_adapter()` — no restart, no downtime. New visitor responses use the updated weights.

### Why this shape

- **Two-topic separation** — audit (`rlhf-pending`) vs. curated (`rlhf-feedback`). Lets the trainer trust its input while keeping a full record of visitor signals
- **Human-in-the-loop curation** — at small data scales, one bad pair has meaningful gradient impact. Manual promotion is the cheapest filter that actually works
- **Streaming online DPO** — most DPO is batched (32-64 pairs per step over a fixed dataset for hours). WiLLi runs **effective batch size 1** with one step per arriving pair. Tradeoff: noisier gradients, immediate responsiveness. Conservative `β=0.1` and small LoRA capacity keep individual steps from causing dramatic drift. For production scale this would buffer into mini-batches of 8–16 pairs
- **Adapter checkpointing with rollback** — every version is saved. If a training step makes the model weird, the admin panel rolls back instantly

### Why DPO over RLHF-PPO

The original RLHF approach (PPO) requires training a separate reward model and running RL with three model copies in memory — operationally complex and finicky. DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) showed that the optimal policy under PPO-with-KL-constraint can be derived analytically from preference data, eliminating the reward model and the RL entirely. You train the policy directly on preferences with a simple log-sigmoid loss. Same expressiveness, dramatically simpler infrastructure. For a single-GPU project this is the only realistic option.

---

## Project Structure

```
streaming-rl-llm/
├── index.html                # React frontend (single file, all tabs)
├── docker-compose.yml        # Redpanda broker config
├── requirements.txt
├── start_redpanda.sh         # WSL Redpanda startup script
├── build_opening_book.py     # Fetch chess.com games → opening_book.json
├── build_sounds.py           # Fetch Lichess sound pack → sounds/
├── opening_book.json         # Will's chess opening repertoire
├── sounds/                   # Chess sound effects (Lichess piano pack)
├── rlhf_service/
│   ├── api.py                # FastAPI backend
│   ├── train_loop.py         # DPO training consumer
│   └── Dockerfile            # GPU training container
├── rag_service/
│   └── ingest.py             # ChromaDB ingestion
└── stream_service/
    └── producer.py           # Redpanda feedback producer
```

Personal automation (`start_willi.bat`, `start_willi_lite.bat`, `.env`, etc.) and runtime artifacts (`willi_adapter_checkpoints/`, `pending_feedback.jsonl`, `training_history.jsonl`) are gitignored — this README walks through manual setup.

---

## Local Setup (Windows + WSL2)

### Prerequisites
- Windows 10/11 with WSL2
- Docker inside WSL2
- Python 3.10+ with venv
- NVIDIA GPU with CUDA drivers
- Cloudflare account (for tunnel, optional for local-only)

### 1. Clone the repo
```bash
git clone https://github.com/wenlong96/streaming-rl-llm.git
cd streaming-rl-llm
```

### 2. Create `.env`
```
KAFKA_BROKER=<your-wsl-ip>:9092
HF_TOKEN=<your-huggingface-token>
ADMIN_PASSWORD=<your-password>
TELEGRAM_TOKEN=<optional>
TELEGRAM_CHAT_ID=<optional>
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

### 5. Start the broker
```bash
# In WSL
bash start_redpanda.sh
```

### 6. Run the services
- **API:** `uvicorn rlhf_service.api:app --port 8000`
- **Frontend:** `python -m http.server 3000` from repo root
- **Trainer:** `python rlhf_service/train_loop.py`
- **Cloudflare tunnel** (optional): `cloudflared tunnel run`

### 7. (Optional) Build chess feature assets
```bash
pip install chess                       # python-chess for opening book
python build_opening_book.py            # pulls your chess.com games
python build_sounds.py --pack piano     # downloads Lichess sound pack
```

---

## Chess Playground

The `/playground` route runs a full chess implementation in the browser:

- Custom React/SVG board with click-to-move
- Stockfish 10.0.2 loaded as a Web Worker (cross-origin loaded via `Blob`-URL workaround)
- Six difficulty modes: **Will** (gold accent — uses Will's actual chess.com opening repertoire and reacts with shitposter commentary), Beginner, Casual, Fair, Intermediate, Hard
- Per-mode game state preservation — switching modes mid-game saves progress; switching back restores
- Promotion picker, color toggle, move history scoresheet, persistent W/L/D record (localStorage)
- Sound effects via Lichess piano pack
- Reactions surface as speech bubbles next to a persistent Will avatar (WiLLi mode only)

The opening book (`opening_book.json`) is a frequency-weighted FEN→UCI map built from Will's chess.com games via `build_opening_book.py`. In WiLLi mode for the first 8 plies, the bot consults the book before falling back to Stockfish — giving the bot Will's actual opening character at low depth.

---

## Acknowledgments

- **Chess sounds** — [Lichess](https://github.com/lichess-org/lila) piano sound pack by Enigmahack, AGPLv3+. See `sounds/NOTICE.txt`
- **Chess pieces** — Cburnett SVG set (public domain)
- **DPO** — [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **chess.js** + **Stockfish.js** for client-side chess engine
- **Qwen 2.5** team for the base model

---

## Secrets & Security

- `.env` is gitignored — never committed
- Personal context (`will_context.txt`, `resume.pdf`) gitignored
- Redpanda runs locally and is not exposed to the internet
- Admin panel password-protected via `sessionStorage`
- Cloudflare Tunnel handles SSL and DDoS protection
- Trained adapter checkpoints and visitor feedback logs gitignored — they're either large binaries or contain visitor input

---

## Why This Project?

WiLLi is a personal project, but technically it's an end-to-end demonstration of the parts of ML engineering that don't usually fit in a notebook: streaming infrastructure, online preference learning, durable feedback queues, hot model reload, and the operational discipline to run all of it on a single consumer GPU behind a residential connection.

Also I hope to have inspired anyone reading this to create something that represents them :)
---

## Author

**Will Lim Wen Long** — Data Scientist · Singapore
[askwilli.dev](https://askwilli.dev) · [GitHub](https://github.com/wenlong96)
