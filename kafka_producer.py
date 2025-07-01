from kafka import KafkaProducer
import json
import time
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def get_kafka_producer(bootstrap_servers=None, retries=5, delay=5):
    bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logging.info(f"Kafka producer connected successfully on attempt {attempt+1}")
            return producer
        except Exception as e:
            logging.error(f"Kafka connection attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Failed to connect to Kafka after {retries} attempts: {str(e)}")
                raise

producer = get_kafka_producer()

def send_kafka_message(topic, message):
    try:
        producer.send(topic, message)
        producer.flush()
        logging.debug(f"Sent Kafka message to {topic}: {message}")
        return True
    except Exception as e:
        logging.error(f"Failed to send Kafka message to {topic}: {str(e)}")
        return False