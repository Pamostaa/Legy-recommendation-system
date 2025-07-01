from kafka import KafkaConsumer
import json
from pymongo.errors import ConnectionFailure
from data_loader import get_mongo_client
from api import orchestrator, save_recommendations, restaurants, products, oid_list, clean_doc
import logging
import time
import os
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://legy:legy123@mongo:27017/leggydb?authSource=admin")

def connect_mongo(uri, retries=5, delay=5):
    for attempt in range(retries):
        try:
            client = get_mongo_client(uri)
            client.admin.command("ping")
            logging.info("MongoDB connection successful")
            return client
        except ConnectionFailure as e:
            logging.error(f"MongoDB connection attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Error: MongoDB connection failed after {retries} attempts: {str(e)}")
                print(f"Error: MongoDB connection failed: {str(e)}")
                exit(1)

def connect_kafka(topic, brokers, group_id, retries=5, delay=5):
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=brokers,
                group_id=group_id,
                auto_offset_reset='earliest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logging.info("Kafka consumer connected successfully")
            return consumer
        except Exception as e:
            logging.error(f"Kafka connection attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Error: Kafka connection failed after {retries} attempts: {str(e)}")
                print(f"Error: Kafka connection failed: {str(e)}")
                exit(1)

mongo_client = connect_mongo(MONGO_URI)
db = mongo_client["leggydb"]
if "user_recommendations" not in db.list_collection_names():
    db.create_collection("user_recommendations")
    logging.info("Created user_recommendations collection")

consumer = connect_kafka('recommendation_requests', KAFKA_BOOTSTRAP_SERVERS, 'recommender-logger')

print("Kafka consumer listening for recommendation_requests...")
for msg in consumer:
    logging.debug(f"Received Kafka message: {msg.value}")
    print("📥 Received Kafka message:", msg.value)
    
    user_id = msg.value.get("user_id")
    
    if not user_id:
        logging.error("No user_id in message")
        print("Error: No user_id in message")
        continue
    
    try:
        logging.debug(f"Generating recommendations for user {user_id}")
        res = orchestrator.get_recommendations(user_id, top_n=5)
        
        if not res or ("Recommendations" not in res and "Products" not in res):
            error_msg = res.get("error", "Empty recommendations")
            logging.error(f"Recommendation failed for user {user_id}: {error_msg}")
            print(f"Error for user {user_id}: {error_msg}")
            continue
        
        if res.get("Recommendations"):
            rest_ids = [rid for _, rid in res["Recommendations"]]
            docs = restaurants.find({"_id": {"$in": oid_list(rest_ids)}})
            payload = {"RecommendedRestaurants": [clean_doc(d) for d in docs]}
            try:
                save_recommendations(user_id, "restaurant", payload)
                logging.debug(f"Stored restaurant recommendations for user {user_id}")
                print(f"Stored restaurant recommendations for user {user_id}")
            except Exception as e:
                logging.error(f"Restaurant save failed for user {user_id}: {str(e)}")
                print(f"Error: Restaurant save failed: {str(e)}")
        
        if res.get("Products"):
            product_recs = []
            for rest_name, product_list in res["Products"].items():
                if not isinstance(product_list, list):
                    logging.error(f"Expected list for products of {rest_name}, got {type(product_list)}")
                    continue
                if len(product_list) != 5:
                    logging.warning(f"Expected 5 products for {rest_name}, got {len(product_list)}")
                rest_id = next((rid for name, rid in res["Recommendations"] if name.lower() == rest_name.lower()), None)
                if not rest_id:
                    logging.warning(f"No restaurant ID found for {rest_name}")
                    continue
                prod_ids = [str(pid) for pid in product_list if pid]
                if prod_ids:
                    docs = products.find({"_id": {"$in": oid_list(prod_ids)}})
                    product_recs.append({
                        "restaurant_id": rest_id,
                        "products": [clean_doc(d) for d in docs]
                    })
                    logging.debug(f"Collected {len(prod_ids)} products for {rest_name} (ID: {rest_id})")
            
            if product_recs:
                payload = {"RecommendedProducts": product_recs}
                try:
                    save_recommendations(user_id, "product", payload)
                    logging.debug(f"Stored product recommendations for user {user_id}")
                    print(f"Stored product recommendations for user {user_id}")
                except Exception as e:
                    logging.error(f"Product save failed for user {user_id}: {str(e)}")
                    print(f"Error: Product save failed: {str(e)}")
            else:
                logging.warning(f"No valid product recommendations for user {user_id}")
                print(f"Warning: No valid product recommendations for user {user_id}")
        
    except Exception as e:
        error_msg = f"Error processing recommendation for user {user_id}: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        print(f"Error processing recommendation for user {user_id}: {str(e)}")