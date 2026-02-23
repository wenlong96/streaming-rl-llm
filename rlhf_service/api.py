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
    retrieval_query = f"Will's experience, projects, and background related to: {request.question}"
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
                "You are Will, a Senior Data Scientist from Singapore. "
                "You speak in first person, are friendly, cheerful, and enthusiastic about AI and data science, "
                "but you are also concise and natural — not over the top. "
                "Never say you are an AI or a language model. "
                "Answer questions about your career, projects, and skills directly and confidently."
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
        for new_text in streamer:
            yield new_text
            time.sleep(0.01)

    return StreamingResponse(generate(), media_type="text/plain")