from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='redpanda:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

prompts = [
    "Explain quantum computing like I'm five.",
    "Write a python function to reverse a list.",
    "What is the capital of France?",
    "Analyze the sentiment of this tweet: I love AI!"
]

while True:
    data = {"prompt": random.choice(prompts)}
    producer.send('user-prompts', value=data)
    print(f"Sent: {data}")
    time.sleep(5) # Send a new prompt every 5 seconds