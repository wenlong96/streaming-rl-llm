import os
import json
import torch
from dotenv import load_dotenv
load_dotenv()
import shutil

os.environ["ACCELERATE_USE_CPU"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from confluent_kafka import Consumer, KafkaError
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datetime import datetime
from datasets import Dataset

# --- Configuration ---
model_id = "Qwen/Qwen2.5-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
ADAPTER_PATH = "./willi_adapter"

adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
if os.path.exists(adapter_config_path):
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    if adapter_config.get("base_model_name_or_path", "") != model_id:
        print(f"Adapter mismatch detected — wiping old adapter and starting fresh.")
        shutil.rmtree(ADAPTER_PATH)
        shutil.rmtree("./willi_checkpoints", ignore_errors=True)

# --- Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.bfloat16  # ← new kwarg
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

# --- Kafka Consumer Setup ---
conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'willi-train',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
consumer.subscribe(['rlhf-feedback'])

print("--- WiLLi DPO Training Loop Active ---")
print(f"Listening for feedback on {KAFKA_BROKER}...")

def truncate_to_tokens(text, max_tokens=200):
    """Truncate text to max_tokens to prevent overflow."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return text

def run_dpo_step(batch):
    
    status_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_status.json")
    with open(status_path, "w") as f:
        json.dump({"status": "training"}, f)
    
    formatted_batch = []
    for item in batch:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        chosen = truncate_to_tokens(item["chosen"], max_tokens=200)
        rejected = truncate_to_tokens(item["rejected"], max_tokens=200)

        formatted_batch.append({
            "prompt": formatted_prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    dataset = Dataset.from_list(formatted_batch)

    training_args = DPOConfig(
        output_dir="./willi_checkpoints",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        learning_rate=5e-4,
        logging_steps=1,
        max_steps=2,
        max_length=1024,
        beta=0.1,        
        truncation_mode="keep_end",
        bf16=True,
    )

    model.config.use_cache = False

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,  
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    dpo_trainer.train()
    
    print("LOG HISTORY:", dpo_trainer.state.log_history)
    
    model.save_pretrained("./willi_adapter")

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_script_path)
    
    train_metrics = dpo_trainer.state.log_history[0] if dpo_trainer.state.log_history else {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_payload = {
        "timestamp": timestamp,
        "prompt_preview": batch[0]['prompt'][:150].replace('\n', ' '),
        "chosen_preview": batch[0]['chosen'][:150].replace('\n', ' '),
        "rejected_preview": batch[0]['rejected'][:150].replace('\n', ' '),
        "loss": train_metrics.get("loss"),
        "grad_norm": train_metrics.get("grad_norm"),
        "rewards_chosen": train_metrics.get("rewards/chosen"),
        "rewards_rejected": train_metrics.get("rewards/rejected"),
        "rewards_margin": train_metrics.get("rewards/margins"),
        "rewards_accuracy": train_metrics.get("rewards/accuracies"),
        "logps_chosen": train_metrics.get("logps/chosen"),
        "logps_rejected": train_metrics.get("logps/rejected"),
    }
    metrics_path = os.path.join(root_dir, "training_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
    except Exception as e:
        print(f"Failed to write metrics: {e}")
    
    log_path = os.path.join(root_dir, "training_logs.txt")

    prompt_text = batch[0]['prompt'][:150].replace('\n', ' ')
    chosen_text = batch[0]['chosen'][:150].replace('\n', ' ')
    rejected_text = batch[0]['rejected'][:150].replace('\n', ' ')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = (
        f"[{timestamp}] TRAINING COMPLETE\n"
        f"Prompt: {prompt_text}...\n"
        f"Chosen: {chosen_text}...\n"
        f"Rejected: {rejected_text}...\n"
        f"{'-'*30}\n"
    )

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"[{timestamp}] Logged update to {log_path}")
    except Exception as e:
        print(f"Failed to write log: {e}")

    print(f"[{timestamp}] Model Updated and Adapter Saved.")

    with open(status_path, "w") as f:
        json.dump({"status": "complete"}, f)

# --- Main Loop ---
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF: continue
            else: print(msg.error()); break

        data = json.loads(msg.value().decode('utf-8'))
        print(f"Received Feedback: {data['prompt'][:50]}...")

        dpo_batch = [{
            "prompt": data['prompt'],
            "chosen": data['chosen'],
            "rejected": data['rejected']
        }]

        run_dpo_step(dpo_batch)

except KeyboardInterrupt:
    pass
finally:
    consumer.close()