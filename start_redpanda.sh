#!/bin/bash
exec > /mnt/c/Users/wl/redpanda_start.log 2>&1
echo "=== START $(date) ==="
echo 1 | sudo -S service docker start
sleep 8
echo "=== Docker started ==="
cd /mnt/c/Users/wl/Desktop/Work/Website/streaming-rl-llm
NEW_IP=$(hostname -I | awk '{print $1}')
echo "=== WSL IP: ${NEW_IP} ==="
sed -i "s/KAFKA_BROKER=.*/KAFKA_BROKER=${NEW_IP}:9092/" .env
sed -i "s|--advertise-kafka-addr internal://redpanda:19092,external://[0-9.]*:9092|--advertise-kafka-addr internal://redpanda:19092,external://${NEW_IP}:9092|" docker-compose.yml
echo "=== Config updated ==="
sudo docker compose down
echo "=== Down complete ==="
sudo docker compose up -d redpanda
echo "=== Up complete ==="
sleep 30
sudo docker ps
sudo docker exec streaming-rl-llm-redpanda-1 rpk topic create rlhf-feedback 2>/dev/null || true
echo "=== DONE ==="
sleep infinity
