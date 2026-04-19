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
import httpx
from stream_service.producer import stream_rlhf_feedback

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
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading base model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map={"": 0} if DEVICE == "cuda" else None
)

# Check adapter compatibility before loading
if os.path.exists(ADAPTER_PATH):
    adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        adapter_base = adapter_config.get("base_model_name_or_path", "")
        if adapter_base != MODEL_ID:
            print(f"Adapter was trained on '{adapter_base}', current model is '{MODEL_ID}'. Skipping adapter.")
            model = base_model
        else:
            print(f"Applying WiLLi's learned DPO personality from {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("No adapter config found. Using base model.")
        model = base_model
else:
    print("No adapter found. Using base model.")
    model = base_model

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

@app.post("/chat")
async def chat_with_clone(request: ChatRequest):
    await notify_me(f"👤 New message on WiLLi!\n\nQuestion: {request.question[:200]}")
    try:
        retrieval_query = f"Will's experience, projects, background, and hobby related to: {request.question}"
        docs = retriever.invoke(retrieval_query)
        
        combined_context = "\n\n".join([doc.page_content for doc in docs])
        
        grounded_user_message = (
            f"Use the context below to answer the question as Will. "
            f"If the context doesn't fully cover the question, answer based on what you know about Will's background. "
            f"Be concise and natural.\n\n"
            f"CONTEXT:\n{combined_context}\n\n"
            f"USER QUESTION: {request.question}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Will, a Data Scientist from Singapore. "
                    "You speak in first person, are friendly, cheerful, and enthusiastic, "
                    "but you are also concise and natural — not over the top. "
                    "Never say you are an AI or a language model. "
                    "Answer questions about you directly and confidently. "
                )
            },
            {"role": "user", "content": grounded_user_message}
        ]
        
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
            # Capture after stream completes
            last_context["prompt"] = request.question
            last_context["answer"] = "".join(full_answer)

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

@app.get("/admin/status")
async def admin_status():
    status_file = os.path.join(os.getcwd(), "training_status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            s = json.load(f)
        return {"status": s.get("status", "idle")}
    return {"status": "idle"}