import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from kafka import KafkaConsumer, KafkaProducer
import json

# 1. Configuration
model_id = "DeepSeek-R1-Distill-Llama-8B" # or "meta-llama/Meta-Llama-3-8B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 2. Load Model (Actor) and Reference Model
# We load in 4-bit to save VRAM
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    load_in_4bit=True,
    peft_config=peft_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Initialize PPO Trainer
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1) # Streaming batch size
ppo_trainer = PPOTrainer(ppo_config, model, ref_model=None, tokenizer=tokenizer)

# 4. Kafka Setup
consumer = KafkaConsumer(
    'user-prompts',
    bootstrap_servers='redpanda:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
producer = KafkaProducer(
    bootstrap_servers='redpanda:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

print("🚀 RLHF Loop Started... Waiting for streams.")

def get_reward(response_text):
    """
    THE CRITIC: This determines if the model gets a cookie or a slap.
    For this demo, we use a simple heuristic: Length and positive sentiment keywords.
    In production, this would be a call to GPT-4 or a local Bert-Reward model.
    """
    score = 0.0
    if len(response_text) > 50: score += 0.5
    if "analysis" in response_text.lower(): score += 1.0
    return torch.tensor([score], device=device)

# 5. The Infinite Training Loop
for message in consumer:
    prompt = message.value['prompt']
    print(f"Received: {prompt}")

    # A. Encode
    query_tensors = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # B. Generate (Rollout)
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=100)
    response_text = tokenizer.decode(response_tensors[0])

    # C. Calculate Reward
    reward_tensors = [get_reward(response_text)]

    # D. PPO Step (The Learning Moment)
    # The model updates its weights based on the reward immediately
    stats = ppo_trainer.step([query_tensors[0]], [response_tensors[0]], reward_tensors)

    # E. Stream Result
    output = {
        "prompt": prompt,
        "response": response_text,
        "reward": reward_tensors[0].item(),
        "ppo_loss": stats['ppo/loss/total']
    }
    producer.send('model-outputs', value=output)
    print(f"Updated Model | Reward: {output['reward']}")