from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
from threading import Thread
import time
from peft import PeftModel 
from dotenv import load_dotenv
load_dotenv()
import json
import uuid
from datetime import datetime
import httpx
from stream_service.producer import stream_rlhf_feedback, stream_pending_feedback

last_context = {"prompt": None, "answer": None}

async def notify_me(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": message}
                )
        except Exception:
            pass

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DB_DIR = "./chroma_db"
ADAPTER_PATH = "./willi_adapter"
CHECKPOINTS_DIR = "./willi_adapter_checkpoints"
MANIFEST_PATH = os.path.join(CHECKPOINTS_DIR, "manifest.json")
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading base model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map={"": 0} if DEVICE == "cuda" else None
)


def load_active_adapter():
    """Load (or reload) the active LoRA adapter from ADAPTER_PATH onto the base model.

    Reads adapter_config.json and verifies base_model_name compatibility before applying.
    Falls back to base model if no compatible adapter is present.
    Returns the resulting model. Safe to call multiple times — used for both startup
    and runtime rollback.
    """
    if not os.path.exists(ADAPTER_PATH):
        print("No adapter found. Using base model.")
        return base_model
    adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print("No adapter config found. Using base model.")
        return base_model
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    adapter_base = adapter_config.get("base_model_name_or_path", "")
    if adapter_base != MODEL_ID:
        print(f"Adapter was trained on '{adapter_base}', current model is '{MODEL_ID}'. Skipping adapter.")
        return base_model
    print(f"Applying WiLLi's learned DPO personality from {ADAPTER_PATH}")
    return PeftModel.from_pretrained(base_model, ADAPTER_PATH)


model = load_active_adapter()

if DEVICE == "cpu":
    model.to(DEVICE)

# Load RAG Database
print("Connecting to Knowledge Base")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if not os.path.exists(DB_DIR):
    raise RuntimeError(f"Database not found at {DB_DIR}")

vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) 
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) 

# Initialize FastAPI
app = FastAPI(title="Will's AI Clone API", version="1.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data format we expect from the frontend UI
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    retrieved_context: list[str]

@app.get("/")
def health_check():
    return {"status": "Online", "model": MODEL_ID, "adapted": os.path.exists(ADAPTER_PATH)}


@app.get("/status")
async def status():
    """Public proof-of-life: chat count, adapter version, last training time.

    Reads existing files; nothing here writes state. Safe to call frequently.
    """
    chat_count = None
    adapter_version = None
    last_trained = None

    # adapter_version = number of completed training events
    log_file = os.path.join(os.getcwd(), "training_logs.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                adapter_version = f.read().count("TRAINING COMPLETE")
        except Exception:
            pass

    # last_trained = relative time of most recent training step
    metrics_file = os.path.join(os.getcwd(), "training_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                m = json.load(f)
            ts = m.get("timestamp")
            if ts:
                t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                diff = (datetime.now() - t).total_seconds()
                if diff < 60:
                    last_trained = "trained just now"
                elif diff < 3600:
                    last_trained = f"trained {int(diff // 60)}m ago"
                elif diff < 86400:
                    last_trained = f"trained {int(diff // 3600)}h ago"
                else:
                    last_trained = f"trained {int(diff // 86400)}d ago"
        except Exception:
            pass

    # chat_count = total feedback votes received (proxy for engagement)
    # Real chat count would need instrumentation in /chat; this is a stand-in.
    pending_file = os.path.join(os.getcwd(), "pending_feedback.jsonl")
    if os.path.exists(pending_file):
        try:
            with open(pending_file, "r", encoding="utf-8") as f:
                chat_count = sum(1 for line in f if line.strip())
        except Exception:
            pass

    return {
        "chat_count": chat_count,
        "adapter_version": adapter_version,
        "last_trained": last_trained,
    }


@app.post("/chat")
async def chat_with_clone(body: dict):
    # Accept either legacy {question} or new {messages: [{role, content}, ...]}
    incoming_msgs = body.get("messages")
    if not incoming_msgs:
        q = body.get("question", "")
        incoming_msgs = [{"role": "user", "content": q}] if q else []

    if not incoming_msgs or incoming_msgs[-1].get("role") != "user":
        return StreamingResponse(iter(["Please ask me something."]), media_type="text/plain")

    current_question = incoming_msgs[-1]["content"]
    prior_turns = incoming_msgs[:-1]  # all turns before the current user message

    await notify_me(f"👤 New message on WiLLi!\n\nQuestion: {current_question[:200]}")
    try:
        docs = retriever.invoke(current_question)
        combined_context = "\n\n".join([doc.page_content for doc in docs])

        # Build the full conversation:
        #   1. Identity system prompt (persistent)
        #   2. RAG context as a separate system note with instructions on HOW to use it
        #   3. Prior turns — clean user/assistant exchanges (NOT re-grounded)
        #   4. Current user turn — clean question (RAG already lives in system)
        # Keeping user/assistant turns clean prevents the model from echoing
        # "based on the context provided" patterns back at the user.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Will Lim Wen Long, a Data Scientist from Singapore. "
                    "You are a real person. Always speak in first person. "
                    "You ARE Will — never refer to 'Will' or 'WiLLi' as if they are someone else. "
                    "When asked about your experiences, projects, or background, answer from your own perspective using 'I', 'my', 'me'. "
                    "Be friendly, warm, and conversational, but also concise and natural. "
                    "Never say you are an AI, language model, or assistant. "
                    "Never say things like 'based on the information provided', 'according to the context', "
                    "or 'the hardest part of building [project] was...' as if narrating someone else's work. "
                    "If asked a follow-up question, infer what it refers to from the recent conversation. "
                    "If you genuinely don't know something about yourself, say 'I'm not sure off the top of my head' rather than making it up. "
                    "Do not use emojis, emoticons, or decorative symbols of any kind. "
                    "Avoid bullet lists and numbered lists unless the user explicitly asks for one — prefer flowing prose. "
                    "Express enthusiasm through word choice, not symbols or formatting."
                )
            },
            {
                "role": "system",
                "content": (
                    "Here is grounding material about your life and work. Use it to answer accurately, "
                    "but do not quote it verbatim or reference it as a source — just speak as yourself and be concise and natural.\n\n"
                    f"{combined_context}"
                )
            },
        ]
        # Append prior turns verbatim — they should be clean exchanges
        for m in prior_turns:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        # Final current turn — bare question, no RAG wrapping
        messages.append({"role": "user", "content": current_question})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        # Initialize the streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Run generation in a separate thread so it doesn't block the stream
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=500,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Generator function for the stream
        def generate():
            full_answer = []
            for new_text in streamer:
                full_answer.append(new_text)
                yield new_text
                time.sleep(0.01)
            # If generation hit max_new_tokens mid-sentence, append a graceful
            # trail marker so the response visually reads as cut off rather
            # than broken. Sentence-ending punctuation = clean stop; anything
            # else means we got cut.
            joined = "".join(full_answer).rstrip()
            if joined and joined[-1] not in '.!?。!?…)"':
                yield " […]"
                joined = joined + " […]"
            # Capture after stream completes
            last_context["prompt"] = current_question
            last_context["answer"] = joined

        return StreamingResponse(generate(), media_type="text/plain")
    except Exception:
        return StreamingResponse(
            iter(["WiLLi is busy right now — please try again in a moment!"]),
            media_type="text/plain"
        )
        
@app.post("/ping_visit")
async def ping_visit():
    await notify_me("👀 Someone just visited askwilli.dev!")
    return {"ok": True}

@app.post("/contact")
async def contact(body: dict):
    contact_info = body.get("contact", "")
    if contact_info:
        await notify_me(f"📬 Contact request!\n\n{contact_info}")
    return {"ok": True}

@app.post("/admin/login")
async def admin_login(body: dict):
    pw = body.get("password", "")
    correct = os.getenv("ADMIN_PASSWORD", "1")
    return {"ok": pw == correct}

@app.get("/admin/metrics")
async def admin_metrics():
    metrics_file = os.path.join(os.getcwd(), "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {}

@app.get("/admin/logs")
async def admin_logs():
    log_file = os.path.join(os.getcwd(), "training_logs.txt")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return {"logs": "".join(lines[:10])}
    return {"logs": "No training events yet."}

@app.post("/admin/preferred")
async def admin_preferred():
    p = last_context.get("prompt")
    a = last_context.get("answer")
    if p and a:
        stream_rlhf_feedback(p, a, a)
        return {"msg": "Positive signal logged ✓"}
    return {"msg": "No recent chat context — ask WiLLi something first."}

@app.post("/admin/correct")
async def admin_correct(body: dict):
    correction = body.get("correction", "")
    p = last_context.get("prompt")
    a = last_context.get("answer")
    if not correction:
        return {"msg": "No correction provided."}
    if not p or not a:
        return {"msg": "No recent chat context — ask WiLLi something first."}
    stream_rlhf_feedback(p, correction, a)
    return {"msg": f"DPO pair pushed to Redpanda ✓"}


# ─────────────────────────────────────────────────────────────────────
# Public feedback (thumbs up/down on chat responses) + admin review
# ─────────────────────────────────────────────────────────────────────

PENDING_FILE = os.path.join(os.getcwd(), "pending_feedback.jsonl")

def _load_pending():
    if not os.path.exists(PENDING_FILE):
        return []
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _save_pending(records):
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def _append_pending(record):
    with open(PENDING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/feedback")
async def feedback(body: dict):
    """Public endpoint — anyone can vote on a chat response."""
    vote = body.get("vote")
    question = (body.get("question") or "").strip()
    answer = (body.get("answer") or "").strip()
    if vote not in ("up", "down"):
        return {"ok": False, "msg": "Invalid vote"}
    if not question or not answer:
        return {"ok": False, "msg": "Missing question or answer"}

    record = {
        "id": uuid.uuid4().hex[:12],
        "ts": datetime.utcnow().isoformat(),
        "vote": vote,
        "question": question,
        "answer": answer,
        "status": "pending",
        "promoted_at": None,
        "promoted_with": None,
        "dismissed_at": None,
    }
    _append_pending(record)

    # Audit trail to Redpanda — best effort, don't fail the request if broker is down
    try:
        audit_record = {k: v for k, v in record.items()
                        if k in ("id", "ts", "vote", "question", "answer")}
        stream_pending_feedback(audit_record)
    except Exception as e:
        print(f"[feedback] Failed to stream to rlhf-pending: {e}")

    # Notify
    emoji = "👍" if vote == "up" else "👎"
    await notify_me(
        f"{emoji} Public feedback on WiLLi!\n\n"
        f"Q: {question[:140]}\n"
        f"A: {answer[:140]}"
    )
    return {"ok": True}


@app.get("/admin/feedback/list")
async def admin_feedback_list(limit: int = 30):
    """Returns pending votes for admin review, newest first."""
    records = _load_pending()
    pending = [r for r in records if r.get("status") == "pending"]
    pending.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return {"items": pending[:limit], "total": len(pending)}


@app.post("/admin/feedback/promote")
async def admin_feedback_promote(body: dict):
    """Promote a pending vote into the rlhf-feedback DPO training stream.

    For 👍: pushes (question, answer, answer) — positive signal, no real DPO update
            but matches existing /admin/preferred behavior.
    For 👎: requires a `correction` — pushes (question, correction, answer)
            so the model learns to prefer the correction over the original answer.
    """
    record_id = body.get("id")
    correction = (body.get("correction") or "").strip()

    records = _load_pending()
    target = next((r for r in records if r.get("id") == record_id), None)
    if not target:
        return {"ok": False, "msg": "Record not found"}
    if target.get("status") != "pending":
        return {"ok": False, "msg": f"Already {target['status']}"}

    question = target["question"]
    answer = target["answer"]
    vote = target["vote"]

    if vote == "up":
        stream_rlhf_feedback(question, answer, answer)
        target["promoted_with"] = "positive_signal"
    else:
        if not correction:
            return {"ok": False, "msg": "Correction required to promote a downvote"}
        stream_rlhf_feedback(question, correction, answer)
        target["promoted_with"] = correction

    target["status"] = "promoted"
    target["promoted_at"] = datetime.utcnow().isoformat()
    _save_pending(records)
    return {"ok": True, "msg": "Pushed to rlhf-feedback ✓"}


@app.post("/admin/feedback/dismiss")
async def admin_feedback_dismiss(body: dict):
    """Mark a pending vote as dismissed without promoting it."""
    record_id = body.get("id")
    records = _load_pending()
    target = next((r for r in records if r.get("id") == record_id), None)
    if not target:
        return {"ok": False, "msg": "Record not found"}
    if target.get("status") != "pending":
        return {"ok": False, "msg": f"Already {target['status']}"}
    target["status"] = "dismissed"
    target["dismissed_at"] = datetime.utcnow().isoformat()
    _save_pending(records)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────
# Adapter checkpoint management — list & activate (rollback)
# ─────────────────────────────────────────────────────────────────────

import shutil

def _read_manifest():
    """Read the checkpoints manifest. Returns {active, history} or empty dict."""
    if not os.path.exists(MANIFEST_PATH):
        return {"active": None, "history": []}
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"active": None, "history": []}

def _write_manifest(data):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(data, f, indent=2)


@app.get("/admin/checkpoints")
async def admin_checkpoints():
    """List available adapter checkpoints with metadata."""
    manifest = _read_manifest()
    active = manifest.get("active")
    ts_map = manifest.get("timestamps", {}) or {}
    items = []
    if not os.path.exists(CHECKPOINTS_DIR):
        return {"items": items, "active": active}

    # If manifest is missing timestamps for existing checkpoints (e.g. created
    # before timestamp tracking was added), backfill them once with current time.
    # Better than showing every old checkpoint as "now" — at least admin sees
    # they were stamped at backfill time, and going forward stamps are correct.
    needs_save = False
    for name in os.listdir(CHECKPOINTS_DIR):
        if name.startswith("v") and os.path.isdir(os.path.join(CHECKPOINTS_DIR, name)):
            if name not in ts_map:
                ts_map[name] = datetime.now().isoformat()
                needs_save = True
    if needs_save:
        manifest["timestamps"] = ts_map
        try:
            _write_manifest(manifest)
        except Exception:
            pass

    for name in sorted(os.listdir(CHECKPOINTS_DIR), reverse=True):
        cp_dir = os.path.join(CHECKPOINTS_DIR, name)
        if not os.path.isdir(cp_dir) or not name.startswith("v"):
            continue
        # Manifest timestamp wins; filesystem mtime is fallback only
        ts = ts_map.get(name)
        if not ts:
            try:
                mtime = os.path.getmtime(cp_dir)
                ts = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                ts = None
        items.append({"version": name, "timestamp": ts, "active": name == active})
    return {"items": items, "active": active}


@app.post("/admin/checkpoints/activate")
async def admin_checkpoints_activate(body: dict):
    """Roll the active adapter back (or forward) to a specific checkpoint version.
    
    Copies the checkpoint contents over willi_adapter/, updates the manifest,
    and reloads the model in-place. Subsequent /chat calls use the new weights.
    """
    global model
    version = body.get("version")
    if not version:
        return {"ok": False, "msg": "Missing version"}

    cp_dir = os.path.join(CHECKPOINTS_DIR, version)
    if not os.path.isdir(cp_dir):
        return {"ok": False, "msg": f"Checkpoint {version} not found"}

    # Copy checkpoint contents over the active adapter directory
    try:
        if os.path.exists(ADAPTER_PATH):
            shutil.rmtree(ADAPTER_PATH)
        shutil.copytree(cp_dir, ADAPTER_PATH)
    except Exception as e:
        return {"ok": False, "msg": f"Failed to swap adapter: {e}"}

    # Update manifest
    manifest = _read_manifest()
    manifest["active"] = version
    _write_manifest(manifest)

    # Reload model with new adapter — in-place reassignment of the module-level model
    try:
        model = load_active_adapter()
        if DEVICE == "cpu":
            model.to(DEVICE)
    except Exception as e:
        return {"ok": False, "msg": f"Adapter swapped on disk but reload failed: {e}"}

    return {"ok": True, "msg": f"Activated {version}", "active": version}


@app.get("/admin/history")
async def admin_history(limit: int = 30):
    """Returns the last N training events as appended to training_history.jsonl."""
    history_file = os.path.join(os.getcwd(), "training_history.jsonl")
    if not os.path.exists(history_file):
        return {"items": []}
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        return {"items": lines[-limit:]}
    except Exception:
        return {"items": []}


@app.get("/admin/status")
async def admin_status():
    status_file = os.path.join(os.getcwd(), "training_status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            s = json.load(f)
        return {"status": s.get("status", "idle")}
    return {"status": "idle"}