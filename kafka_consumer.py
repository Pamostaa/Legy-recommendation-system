from kafka import KafkaConsumer
import json
from pymongo.errors import ConnectionFailure
from data_loader import get_mongo_client  # Use data_loader for MongoDB client
from api import orchestrator, save_recs, restaurants, products, oid_list, clean_doc  # Import from api.py
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# MongoDB connection
try:
    mongo_client = get_mongo_client()
    db = mongo_client["recommender_db"]
    # Ensure user_recommendations collection exists
    if "user_recommendations" not in db.list_collection_names():
        db.create_collection("user_recommendations")
        logging.info("Created user_recommendations collection")
    # Test connection
    mongo_client.admin.command("ping")
    logging.info("MongoDB connection successful")
except ConnectionFailure as e:
    logging.error(f"MongoDB connection failed: {str(e)}")
    print(f"Error: MongoDB connection failed: {str(e)}")
    exit(1)

consumer = KafkaConsumer(
    'recommendation_requests',
    bootstrap_servers='localhost:9092',
    group_id='recommender-logger',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

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
        # Generate recommendations
        logging.debug(f"Generating recommendations for user {user_id}")
        res = orchestrator.get_recommendations(user_id, top_n=5)
        
        if not res or ("Recommendations" not in res and "Products" not in res):
            error_msg = res.get("error", "Empty recommendations")
            logging.error(f"Recommendation failed for user {user_id}: {error_msg}")
            print(f"Error for user {user_id}: {error_msg}")
            continue
        
        # Process restaurant recommendations
        if res.get("Recommendations"):
            rest_ids = [rid for _, rid in res["Recommendations"]]
            docs = restaurants.find({"_id": {"$in": oid_list(rest_ids)}})
            payload = {"RecommendedRestaurants": [clean_doc(d) for d in docs]}
            try:
                save_recs(user_id, "restaurant", payload)
                logging.debug(f"Stored restaurant recommendations for user {user_id}")
                print(f"Stored restaurant recommendations for user {user_id}")
            except Exception as e:
                logging.error(f"Restaurant save failed for user {user_id}: {str(e)}")
                print(f"Error: Restaurant save failed: {str(e)}")
        
        # Process product recommendations
        if res.get("Products"):
            prod_ids = []
            for rest_name, product_list in res["Products"].items():
                if not isinstance(product_list, list):
                    logging.error(f"Expected list for products of {rest_name}, got {type(product_list)}")
                    continue
                prod_ids.extend([str(pid) for pid in product_list if pid])
            
            if prod_ids:
                docs = products.find({"_id": {"$in": oid_list(prod_ids)}})
                payload = {"RecommendedProducts": [clean_doc(d) for d in docs]}
                try:
                    save_recs(user_id, "product", payload)
                    logging.debug(f"Stored product recommendations for user {user_id}")
                    print(f"Stored product recommendations for user {user_id}")
                except Exception as e:
                    logging.error(f"Product save failed for user {user_id}: {str(e)}")
                    print(f"Error: Product save failed: {str(e)}")
            else:
                logging.warning(f"No valid product IDs for user {user_id}")
                print(f"Warning: No valid product IDs for user {user_id}")
        
    except Exception as e:
        logging.error(f"Error processing recommendation for user {user_id}: {str(e)}")
        print(f"Error processing recommendation for user {user_id}: {str(e)}")