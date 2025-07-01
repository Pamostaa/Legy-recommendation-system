from flask import Flask, jsonify, request
from flask_cors import CORS
import yaml
import pandas as pd
import pickle
from datetime import datetime
from bson import ObjectId
import logging
import os
import time
from pymongo.errors import ConnectionFailure
from kafka import KafkaProducer
import json

# ========== App components ==========
from data_loader import get_mongo_client, get_collections, reload_users, resolve_user
from model_handler import BERTModelHandler
from collaborative_engine import CollaborativeEngine
from content_engine import ContentEngine
from fallback_engine import FallbackEngine
from orchestrator import RecommendationOrchestrator
from user_repository import UserRepository
from product_repository import ProductRepository
from order_repository import OrderRepository
from drive_utils import ensure_model_files

# ========== Config ==========
with open("config.yaml") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model_path"]
VECTORS_PATH = config["vectors_path"]
ALPHA = config.get("alpha", 0.01)
LAMBDA_MMR = config.get("lambda_mmr", 0.7)
LIKE_WEIGHT = config.get("feedback_weights", {}).get("like_weight", 0.1)
DISLIKE_PENALTY = config.get("feedback_weights", {}).get("dislike_penalty", 0.2)
KAFKA_BROKERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")

# ========== Ensure model files are available ==========
ensure_model_files()

app = Flask(__name__)
CORS(app)

# ========== Logging ==========
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ========== Mongo ==========
def connect_mongo(retries=5, delay=5):
    for attempt in range(retries):
        try:
            client = get_mongo_client()
            client.admin.command("ping")
            logging.info("MongoDB connection successful")
            logging.debug(f"Databases available: {client.list_database_names()}")
            return client
        except ConnectionFailure as e:
            logging.error(f"MongoDB connection attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Error: MongoDB connection failed after {retries} attempts: {str(e)}")
                raise

# ========== Kafka ==========
def get_kafka_producer(bootstrap_servers, retries=5, delay=5):
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logging.info("Kafka producer connected successfully")
            return producer
        except Exception as e:
            logging.error(f"Kafka connection attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Error: Kafka connection failed after {retries} attempts: {str(e)}")
                raise

client = connect_mongo()
collections = get_collections(client)
db = client["leggydb"]
logging.debug(f"Using database: {db.name}")
user_recs = db["user_recommendations"]
restaurants = collections["restaurants"]
products = collections["products"]
users = collections["users"]
avis_restaurant = collections["avis-restaurant"]
orders = collections["orders"]
categories = collections["categories"]
restaurant_reactions = collections["restaurant_reactions"]
user_neighbors = collections["user_neighbors"]

# ========== Data ==========
users_df, name_to_id, id_to_name = reload_users(collections["users"])
labels = [
    "cares_about_food_quality", "cares_about_service_speed",
    "cares_about_price", "cares_about_cleanliness",
]

# ========== Initialize model handler ==========
model_handler = BERTModelHandler(MODEL_PATH, VECTORS_PATH)
id_to_vector = model_handler.id_to_vector

def load_vectors_chunked(path, chunk_size=10000):
    try:
        with open(path, "rb") as f:
            vectors = pickle.load(f)
        items = list(vectors.items())
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
        logging.debug(f"Loaded {len(items)} vectors from {path}")
    except Exception as e:
        logging.error(f"Error loading vectors from {path}: {str(e)}", exc_info=True)
        raise

logging.debug(f"Loaded {len(id_to_vector)} user vectors")

restaurant_name_to_id = {
    doc.get("nom", "").strip().lower(): str(doc["_id"])
    for doc in restaurants.find({}, {"nom": 1})
}
logging.debug(f"Restaurant name to ID mapping (sample): {list(restaurant_name_to_id.items())[:5]}")

user_repo = UserRepository(collections["users"])
product_repo = ProductRepository(collections["products"])
order_repo = OrderRepository(collections["orders"])
ratings_df = pd.DataFrame(list(collections["avis-restaurant"].find({})))
if not ratings_df.empty and 'score' in ratings_df.columns:
    ratings_df['score'] = pd.to_numeric(ratings_df['score'].str.replace(',', '.'), errors='coerce')
    ratings_df['Rating'] = ratings_df['score']
    ratings_df = ratings_df.rename(columns={"restaurantId": "RestaurantId"})
logging.debug(f"Ratings DataFrame shape: {ratings_df.shape}")
logging.debug(f"Ratings columns after rename: {list(ratings_df.columns)}")

# ========== Engines ==========
collab_engine = CollaborativeEngine(
    model_handler, collections, id_to_vector,
    name_to_id, id_to_name, labels,
    restaurant_name_to_id, order_repo, product_repo,
    alpha=ALPHA,
)
content_engine = ContentEngine(product_repo, order_repo, collections["categories"], weights=tuple(config["weights"]))
fallback_engine = FallbackEngine(ratings_df, collections=collections)

orchestrator = RecommendationOrchestrator(
    collab_engine, content_engine, fallback_engine,
    id_to_name, id_to_vector, labels,
    restaurant_name_to_id, collections,
    lambda_mmr=LAMBDA_MMR,
    like_weight=LIKE_WEIGHT,
    dislike_penalty=DISLIKE_PENALTY,
)

# ========== Kafka Producer ==========
producer = get_kafka_producer(KAFKA_BROKERS)

def send_kafka_message(topic, message):
    try:
        producer.send(topic, message)
        producer.flush()
        logging.debug(f"Sent Kafka message to {topic}: {message}")
    except Exception as e:
        logging.error(f"Failed to send Kafka message: {str(e)}", exc_info=True)

# ========== Helpers ==========
def oid_list(ids):
    try:
        return [ObjectId(x) for x in ids if x and ObjectId.is_valid(str(x))]
    except Exception as e:
        logging.error(f"Error in oid_list: {str(e)}", exc_info=True)
        return []

def clean_doc(doc):
    try:
        doc_copy = doc.copy()
        doc_copy["_id"] = str(doc_copy["_id"])
        if "restaurantId" in doc_copy:
            doc_copy["restaurantId"] = str(doc_copy["restaurantId"])
        if "restaurant_Id" in doc_copy:
            doc_copy["restaurant_Id"] = str(doc_copy["restaurant_Id"])
        return doc_copy
    except Exception as e:
        logging.error(f"Error in clean_doc: {str(e)}", exc_info=True)
        return None

def save_recommendations(user_id: str, rec_type: str, payload: dict):
    try:
        user_recs.update_one(
            {"user_id": user_id, "type": rec_type},
            {"$set": {"recommendations": payload, "created_at": datetime.utcnow()}},
            upsert=True,
        )
        logging.debug(f"Saved recommendations for user {user_id}, type {rec_type}")
    except Exception as e:
        logging.error(f"Failed to save recommendations for user {user_id}: {str(e)}", exc_info=True)
        raise

# ========== Routes ==========
@app.route("/recommendations/restaurants")
def generate_restaurants():
    user_input = request.args.get("user_id")
    if not user_input:
        logging.error("Missing user_id in request")
        return jsonify({"error": "Missing user_id"}), 400

    user_id, username = resolve_user(users_df, name_to_id, id_to_name, user_input)
    logging.debug(f"Resolved user_id: '{user_id}', username: '{username}'")
    if not user_id or not username:
        logging.error(f"User not found for input: '{user_input}'")
        return jsonify({"error": f"User not found: {user_input}"}), 404

    try:
        logging.debug(f"Calling orchestrator.get_recommendations for user_id: '{user_id}'")
        res = orchestrator.get_recommendations(user_id, top_n=5)
        logging.debug(f"Orchestrator response for user {user_id}: {res}")
        if not res or not res.get("Recommendations"):
            logging.warning(f"No recommendations generated for user {user_id}")
            return jsonify({"error": f"No recommendations for user {user_id}"}), 404

        rest_ids = [rid for _, rid in res["Recommendations"] if ObjectId.is_valid(rid)]
        logging.debug(f"Restaurant IDs: {rest_ids}")
        if not rest_ids:
            logging.warning(f"No valid restaurant IDs for user {user_id}")
            return jsonify({"RecommendedRestaurants": []}), 200

        docs = restaurants.find({"_id": {"$in": oid_list(rest_ids)}})
        payload = {"RecommendedRestaurants": [d for d in (clean_doc(doc) for doc in docs) if d is not None]}
        logging.debug(f"Payload: {payload}")
        
        save_recommendations(user_id, "restaurant", payload)
        try:
            send_kafka_message("recommendation_requests", {
                "user_id": user_id,
                "type": "restaurant",
                "result": payload
            })
        except Exception as e:
            logging.error(f"Kafka send failed for user {user_id}: {str(e)}", exc_info=True)
        
        return jsonify(payload), 200
    except Exception as e:
        logging.error(f"Error generating restaurant recommendations for user {user_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/recommendations/products")
def generate_products():
    user_input = request.args.get("user_id")
    if not user_input:
        logging.error("Missing user_id in request")
        return jsonify({"error": "Missing user_id"}), 400

    user_id, username = resolve_user(users_df, name_to_id, id_to_name, user_input)
    logging.debug(f"Resolved user_id: '{user_id}', username: '{username}'")
    if not user_id or not username:
        logging.error(f"User not found for input: '{user_input}'")
        return jsonify({"error": f"User not found: {user_input}"}), 404

    try:
        logging.debug(f"Calling orchestrator.get_recommendations for user_id: '{user_id}'")
        res = orchestrator.get_recommendations(user_id, top_n=5)
        logging.debug(f"Products response for user {user_id}: {res}")
        if not res or not res.get("Products"):
            logging.warning(f"No product recommendations for user {user_id}")
            return jsonify({"error": f"No recommendations for user {user_id}"}), 404

        product_recs = []
        for rest_name, product_list in res["Products"].items():
            try:
                if not isinstance(product_list, list):
                    logging.error(f"Expected list for products of {rest_name}, got {type(product_list)}")
                    continue
                if len(product_list) == 0:
                    logging.warning(f"Empty product list for {rest_name}")
                    continue
                if len(product_list) != 5:
                    logging.warning(f"Expected 5 products for {rest_name}, got {len(product_list)}")
                rest_id = next((rid for name, rid in res["Recommendations"] if name.lower() == rest_name.lower()), None)
                if not rest_id:
                    logging.warning(f"No restaurant ID found for {rest_name}")
                    continue
                prod_ids = [str(pid) for pid in product_list if pid and ObjectId.is_valid(str(pid))]
                if not prod_ids:
                    logging.warning(f"No valid product IDs for {rest_name}: {product_list}")
                    continue
                logging.debug(f"Querying products for IDs: {prod_ids}")
                docs = list(products.find({"_id": {"$in": oid_list(prod_ids)}}))
                if not docs:
                    logging.warning(f"No products found for IDs: {prod_ids}")
                    continue
                cleaned_docs = [d for d in (clean_doc(doc) for doc in docs) if d is not None]
                if not cleaned_docs:
                    logging.warning(f"No valid cleaned documents for {rest_name}")
                    continue
                product_recs.append({
                    "restaurant_id": rest_id,
                    "products": cleaned_docs
                })
                logging.debug(f"Collected {len(cleaned_docs)} products for {rest_name} (ID: {rest_id})")
            except Exception as e:
                logging.error(f"Error processing products for {rest_name}: {str(e)}", exc_info=True)
                continue

        if not product_recs:
            logging.warning(f"No valid product recommendations for user {user_id}")
            return jsonify({"error": "No valid product recommendations available"}), 404

        payload = {"RecommendedProducts": product_recs}
        logging.debug(f"Product payload: {payload}")
        
        save_recommendations(user_id, "product", payload)
        try:
            send_kafka_message("recommendation_requests", {
                "user_id": user_id,
                "type": "product",
                "result": payload
            })
        except Exception as e:
            logging.error(f"Kafka send failed for user {user_id}: {str(e)}", exc_info=True)

        return jsonify(payload), 200
    except Exception as e:
        logging.error(f"Error generating product recommendations for user {user_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/stored/recommendations/restaurants")
def stored_restaurants():
    user_id = request.args.get("user_id")
    if not user_id:
        logging.error("Missing user_id in request")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        logging.debug(f"Querying user_recommendations for user_id: '{user_id}', type: restaurant")
        logging.debug(f"Database: {db.name}, Collection: user_recommendations")
        doc = user_recs.find_one({"user_id": user_id.strip(), "type": "restaurant"})
        if not doc:
            logging.warning(f"No stored restaurant recommendations found for user_id: '{user_id}'")
            logging.debug(f"Available user_ids in user_recommendations: {[d['user_id'] for d in user_recs.find({'type': 'restaurant'}, {'user_id': 1})]}")
            return jsonify({"error": "No stored restaurant recommendations"}), 404
        logging.debug(f"Found document: {doc}")
        recommendations = doc.get("recommendations", {})
        logging.debug(f"Recommendations for user_id: '{user_id}': {recommendations}")
        if not recommendations.get("RecommendedRestaurants"):
            logging.warning(f"No RecommendedRestaurants in document for user_id: '{user_id}'")
            return jsonify({"error": "No stored restaurant recommendations"}), 404
        return jsonify(recommendations), 200
    except Exception as e:
        logging.error(f"Error retrieving stored restaurant recommendations for user_id: '{user_id}': {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/stored/recommendations/products")
def stored_products():
    user_id = request.args.get("user_id")
    if not user_id:
        logging.error("Missing user_id in request")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        logging.debug(f"Querying user_recommendations for user_id: '{user_id}', type: product")
        logging.debug(f"Database: {db.name}, Collection: user_recommendations")
        doc = user_recs.find_one({"user_id": user_id.strip(), "type": "product"})
        if not doc:
            logging.warning(f"No stored product recommendations for user_id: '{user_id}'")
            logging.debug(f"Available user_ids in user_recommendations: {[d['user_id'] for d in user_recs.find({'type': 'product'}, {'user_id': 1})]}")
            return jsonify({"error": "No stored product recommendations"}), 404
        logging.debug(f"Found document: {doc}")
        recommendations = doc.get("recommendations", {})
        logging.debug(f"Recommendations for user_id: '{user_id}': {recommendations}")
        if not recommendations.get("RecommendedProducts"):
            logging.warning(f"No RecommendedProducts in document for user_id: '{user_id}'")
            return jsonify({"error": "No stored product recommendations"}), 404
        return jsonify(recommendations), 200
    except Exception as e:
        logging.error(f"Error retrieving stored product recommendations for user_id: '{user_id}': {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health")
def health():
    try:
        client.admin.command("ping")
        logging.debug("Health check: MongoDB ping successful")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    logging.info("API running on http://0.0.0.0:8000")
    app.run("0.0.0.0", 8000, debug=False)