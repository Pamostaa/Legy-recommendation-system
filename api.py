from flask import Flask, jsonify, request
from flask_cors import CORS
import yaml
import pandas as pd
import pickle

from data_loader import get_mongo_client, get_collections, reload_users
from model_handler import BERTModelHandler
from collaborative_engine import CollaborativeEngine
from content_engine import ContentEngine
from fallback_engine import FallbackEngine
from orchestrator import RecommendationOrchestrator
from mongo_utils import save_recommendations_to_mongo
from user_repository import UserRepository
from product_repository import ProductRepository
from order_repository import OrderRepository

# === Load config ===
with open('config.yaml') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
CORS(app)

MODEL_PATH = config['model_path']
VECTORS_PATH = config['vectors_path']
ALPHA = config.get('alpha', 0.01)

# === Load weights ===
weights_config = config.get('weights', {})
w1 = weights_config.get('ingredient_similarity', 0.3)
w2 = weights_config.get('price_proximity', 0.3)
w3 = weights_config.get('rating', 0.2)
w4 = weights_config.get('sentiment', 0.2)

# === Load vectors ===
with open(VECTORS_PATH, "rb") as f:
    id_to_vector = pickle.load(f)

# === Mongo setup ===
client = get_mongo_client()
collections = get_collections(client)
users_df, name_to_id, id_to_name = reload_users(collections["users"])
labels = ["cares_about_food_quality", "cares_about_service_speed", "cares_about_price", "cares_about_cleanliness"]
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
content_engine = ContentEngine(product_repo, order_repo, weights=(w1, w2, w3, w4))
fallback_engine = FallbackEngine(ratings_df)
orchestrator = RecommendationOrchestrator(
    collab_engine, content_engine, fallback_engine, id_to_name, id_to_vector, labels,
    restaurant_name_to_id, collections
)

@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    result = orchestrator.get_recommendations(user_id, top_n=5)
    if result:
        save_recommendations_to_mongo(collections["db"], user_id, result)
        return jsonify(result), 200
    else:
        return jsonify({"error": f"No recommendations for user {user_id}"}), 404

@app.route("/users", methods=["GET"])
def list_users():
    users = collections["db"]["user_recommendations"].find({}, {"user_id": 1, "_id": 0})
    return jsonify([u["user_id"] for u in users])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
