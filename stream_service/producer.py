import os
import json
from confluent_kafka import Producer

# Local vs Docker
BOOTSTRAP_SERVERS = os.getenv('KAFKA_BROKER', 'localhost:9092')

# Reuse a single producer across calls — cheaper than reconnecting per request
_producer = None

def get_producer():
    global _producer
    if _producer is None:
        _producer = Producer({
            'bootstrap.servers': BOOTSTRAP_SERVERS,
            'client.id': 'willi-producer',
        })
    return _producer

def _delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def stream_rlhf_feedback(prompt, chosen, rejected):
    """Push a DPO training pair to rlhf-feedback (consumed by train_loop.py)."""
    payload = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    producer = get_producer()
    producer.produce(
        'rlhf-feedback',
        value=json.dumps(payload).encode('utf-8'),
        callback=_delivery_report,
    )
    producer.flush()

def stream_pending_feedback(record):
    """Push a public vote to rlhf-pending as an audit trail.

    Record shape: {id, ts, vote, question, answer}.
    The admin review queue lives in pending_feedback.jsonl;
    this topic is the immutable record of every vote ever cast.
    """
    producer = get_producer()
    producer.produce(
        'rlhf-pending',
        value=json.dumps(record).encode('utf-8'),
        callback=_delivery_report,
    )
    producer.flush()
