from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'recommendation_requests',
    bootstrap_servers='localhost:9092',
    group_id='recommender-logger',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Kafka consumer listening for recommendation_requests...")
for msg in consumer:
    print("📥 Received Kafka message:", msg.value)
