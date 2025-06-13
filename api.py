from flask import Flask, jsonify, request
from flask_cors import CORS
import yaml, pandas as pd, pickle
from datetime import datetime
from bson import ObjectId

# ========== App components ==========
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

# ========== Config ==========
with open("config.yaml") as f:
    config = yaml.safe_load(f)

MODEL_PATH      = config["model_path"]
VECTORS_PATH    = config["vectors_path"]
ALPHA           = config.get("alpha", 0.01)
LAMBDA_MMR      = config.get("lambda_mmr", 0.7)
LIKE_WEIGHT     = config.get("feedback_weights", {}).get("like_weight", 0.1)
DISLIKE_PENALTY = config.get("feedback_weights", {}).get("dislike_penalty", 0.2)

app = Flask(__name__)
CORS(app)

# ========== Mongo ==========
client       = get_mongo_client()
collections  = get_collections(client)
db           = client["recommender_db"]
user_recs    = db["user_recommendations"]
restaurants  = collections["restaurants"]
products     = collections["products"]

# ========== Data ==========
users_df, name_to_id, id_to_name = reload_users(collections["users"])
labels = [
    "cares_about_food_quality", "cares_about_service_speed",
    "cares_about_price", "cares_about_cleanliness",
]

with open(VECTORS_PATH, "rb") as f:
    id_to_vector = pickle.load(f)

restaurant_name_to_id = {
    doc.get("nom", "").strip().lower(): str(doc["_id"])
    for doc in restaurants.find({}, {"nom": 1})
}

user_repo    = UserRepository(collections["users"])
product_repo = ProductRepository(products)
order_repo   = OrderRepository(collections["orders"])
ratings_df   = pd.DataFrame(list(collections["reviews"].find({})))

# ========== Engines ==========
model_handler   = BERTModelHandler(MODEL_PATH, VECTORS_PATH)
collab_engine   = CollaborativeEngine(
    model_handler, collections, id_to_vector,
    name_to_id, id_to_name, labels,
    restaurant_name_to_id, order_repo, product_repo,
    alpha=ALPHA,
)
content_engine  = ContentEngine(product_repo, order_repo)
fallback_engine = FallbackEngine(ratings_df)

orchestrator = RecommendationOrchestrator(
    collab_engine, content_engine, fallback_engine,
    id_to_name, id_to_vector, labels,
    restaurant_name_to_id, collections,
    lambda_mmr=LAMBDA_MMR,
    like_weight=LIKE_WEIGHT,
    dislike_penalty=DISLIKE_PENALTY,
)

# ========== Helpers ==========
def oid_list(ids):
    return [ObjectId(x) for x in ids if ObjectId.is_valid(x)]

def clean_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def save_recs(user_id: str, rec_type: str, payload: dict):
    user_recs.update_one(
        {"user_id": user_id, "type": rec_type},
        {"$set": {"recommendations": payload, "created_at": datetime.utcnow()}},
        upsert=True,
    )

# ========== ROUTES ==========

@app.route("/recommendations/restaurants")
def generate_restaurants():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    res = orchestrator.get_recommendations(user_id, top_n=5)
    if not res or not res.get("Recommendations"):
        return jsonify({"error": f"No recommendations for user {user_id}"}), 404

    rest_ids = [rid for _, rid in res["Recommendations"]]
    docs = restaurants.find({"_id": {"$in": oid_list(rest_ids)}})  # all fields

    payload = {"RecommendedRestaurants": [clean_doc(d) for d in docs]}
    save_recs(user_id, "restaurant", payload)
    return jsonify(payload), 200


@app.route("/recommendations/products")
def generate_products():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    res = orchestrator.get_recommendations(user_id, top_n=5)
    if not res or not res.get("Products"):
        return jsonify({"error": f"No recommendations for user {user_id}"}), 404

    # collect product IDs from orchestrator result
    prod_ids = []
    for lst in res["Products"].values():
        prod_ids += [p.get("id") or p.get("_id") for p in lst]

    # fetch EVERY field for each product
    docs = products.find({"_id": {"$in": oid_list(prod_ids)}})  # no projection

    payload = {"RecommendedProducts": [clean_doc(d) for d in docs]}
    save_recs(user_id, "product", payload)

    try:
        send_kafka_message("recommendation_requests", {
            "user_id": user_id,
            "type": "product",
            "result": payload
        })
    except Exception as e:
        print("[WARN] Kafka send failed:", e)

    return jsonify(payload), 200


@app.route("/stored/recommendations/restaurants")
def stored_restaurants():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    doc = user_recs.find_one({"user_id": user_id, "type": "restaurant"})
    if not doc:
        return jsonify({"error": "No stored restaurant recommendations"}), 404
    return jsonify(doc["recommendations"]), 200


@app.route("/stored/recommendations/products")
def stored_products():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    doc = user_recs.find_one({"user_id": user_id, "type": "product"})
    if not doc:
        return jsonify({"error": "No stored product recommendations"}), 404
    return jsonify(doc["recommendations"]), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print("[INFO] API running on http://localhost:8000")
    app.run("0.0.0.0", 8000, debug=True)
