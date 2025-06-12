from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_kafka_message(topic, message):
    try:
        producer.send(topic, value=message)
        producer.flush()
        print(f"[KAFKA] Sent to topic '{topic}': {message}")
    except Exception as e:
        print(f"[KAFKA ERROR] {e}")
