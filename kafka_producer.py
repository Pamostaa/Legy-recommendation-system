from kafka import KafkaProducer
import json
import time

def get_kafka_producer(bootstrap_servers='kafka:9092', retries=5, delay=5):
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("[INFO] Connected to Kafka")
            return producer
        except Exception as e:
            print(f"[WARN] Kafka connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise Exception("Failed to connect to Kafka after retries")

producer = get_kafka_producer()

def send_kafka_message(topic, message):
    try:
        producer.send(topic, message)
        producer.flush()
    except Exception as e:
        print(f"[ERROR] Failed to send Kafka message: {e}")