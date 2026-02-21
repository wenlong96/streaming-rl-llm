FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install transformers trl peft accelerate datasets kafka-python vllm bitsandbytes

COPY . .

CMD ["python", "train_loop.py"]