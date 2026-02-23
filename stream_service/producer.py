import os
import json
from confluent_kafka import Producer

# Local vs Docker
BOOTSTRAP_SERVERS = os.getenv('KAFKA_BROKER', 'localhost:9092')

def get_producer():
    conf = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'client.id': 'willi-producer',
    }
    return Producer(conf)

def stream_rlhf_feedback(prompt, chosen, rejected):
    producer = get_producer()
    
    payload = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }
    
    def delivery_report(err, msg):
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    producer.produce(
        'rlhf-feedback', 
        value=json.dumps(payload).encode('utf-8'), 
        callback=delivery_report
    )
    
    producer.flush()