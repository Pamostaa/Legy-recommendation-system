from flask import Flask, jsonify, request
from flask_cors import CORS
import yaml
import pandas as pd
import pickle
import re
from bson import ObjectId

# === NLP and utilities ===
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from fuzzywuzzy import fuzz

# === App components ===
from data_loader import get_mongo_client, get_collections, reload_users
from model_handler import BERTModelHandler
from collaborative_engine import CollaborativeEngine
from content_engine import ContentEngine
from fallback_engine import FallbackEngine
from orchestrator import RecommendationOrchestrator
from user_repository import UserRepository
from product_repository import ProductRepository
from order_repository import OrderRepository
from kafka_producer import send_kafka_message

# === Load config ===
with open('config.yaml') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
CORS(app)

# === Config variables ===
MODEL_PATH = config['model_path']
VECTORS_PATH = config['vectors_path']
ALPHA = config.get('alpha', 0.01)
LAMBDA_MMR = config.get('lambda_mmr', 0.7)
LIKE_WEIGHT = config.get('feedback_weights', {}).get('like_weight', 0.1)
DISLIKE_PENALTY = config.get('feedback_weights', {}).get('dislike_penalty', 0.2)

# === Load vectors ===
with open(VECTORS_PATH, "rb") as f:
    id_to_vector = pickle.load(f)

# === Mongo setup ===
client = get_mongo_client()
collections = get_collections(client)

# Debug: show loaded collections
print("[DEBUG] Loaded collections:", collections.keys())

users_df, name_to_id, id_to_name = reload_users(collections["users"])
labels = ["cares_about_food_quality", "cares_about_service_speed", "cares_about_price", "cares_about_cleanliness"]

# ✅ Use the fixed key "restaurants" here
restaurant_name_to_id = {
    r.get("nom").strip().lower(): str(r["_id"])
    for r in collections["restaurants"].find()
    if "nom" in r and "_id" in r
}

# === Initialize repositories ===
user_repo = UserRepository(collections["users"])
product_repo = ProductRepository(collections["products"])
order_repo = OrderRepository(collections["orders"])

# === Load ratings from Reviews collection ===
reviews = list(collections["reviews"].find({}))
ratings_df = pd.DataFrame(reviews)

# === Initialize engines ===
model_handler = BERTModelHandler(MODEL_PATH, VECTORS_PATH)
collab_engine = CollaborativeEngine(
    model_handler, collections, id_to_vector, name_to_id, id_to_name, labels,
    restaurant_name_to_id, order_repo, product_repo, alpha=ALPHA
)
content_engine = ContentEngine(product_repo, order_repo)
fallback_engine = FallbackEngine(ratings_df)
orchestrator = RecommendationOrchestrator(
    collab_engine, content_engine, fallback_engine,
    id_to_name, id_to_vector, labels, restaurant_name_to_id, collections,
    lambda_mmr=LAMBDA_MMR,
    like_weight=LIKE_WEIGHT,
    dislike_penalty=DISLIKE_PENALTY
)

# === ROUTES ===

@app.route("/recommendations", methods=["GET"])
def get_full_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    result = orchestrator.get_recommendations(user_id, top_n=5)
    if result:
        return jsonify(result), 200
    else:
        return jsonify({"error": f"No recommendations for user {user_id}"}), 404


@app.route("/recommendations/restaurants", methods=["GET"])
def get_restaurant_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    result = orchestrator.get_recommendations(user_id, top_n=5)
    if result:
        return jsonify({
            "RecommendedRestaurants": result["Recommendations"]
        }), 200
    else:
        return jsonify({"error": f"No recommendations for user {user_id}"}), 404


@app.route("/recommendations/products", methods=["GET"])
def get_product_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    result = orchestrator.get_recommendations(user_id, top_n=5)

    # 📤 Send Kafka event
    send_kafka_message("recommendation_requests", {
        "user_id": user_id,
        "type": "product",
        "result": result if result else {}
    })

    if result:
        recommended_products = []
        for rest, products in result["Products"].items():
            if isinstance(products, list):
                for p in products:
                    recommended_products.append({
                        "restaurant": rest,
                        "name": p.get("name"),
                        "price": p.get("price"),
                        "category": p.get("categorieName", "")
                    })
        return jsonify({
            "RecommendedProducts": recommended_products
        }), 200
    else:
        return jsonify({"error": f"No recommendations for user {user_id}"}), 404

# === Start the app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
