import yaml
import pandas as pd
import pickle
from pprint import pprint

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
from drive_utils import ensure_model_files

# === Load config ===
with open('config.yaml') as f:
    config = yaml.safe_load(f)

MODEL_PATH = config['model_path']
VECTORS_PATH = config['vectors_path']
ALPHA = config.get('alpha', 0.01)

# === Ensure model files are available ===
ensure_model_files()

# === Initialize model handler ===
model_handler = BERTModelHandler(MODEL_PATH, VECTORS_PATH)
id_to_vector = model_handler.id_to_vector

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
reviews = list(collections["avis-restaurant"].find({}))
ratings_df = pd.DataFrame(reviews)
if not ratings_df.empty and 'score' in ratings_df.columns:
    ratings_df['score'] = pd.to_numeric(ratings_df['score'].str.replace(',', '.'), errors='coerce')
    ratings_df['Rating'] = ratings_df['score']

# === Initialize engines ===
model_handler = BERTModelHandler(MODEL_PATH, VECTORS_PATH)
collab_engine = CollaborativeEngine(
    model_handler, collections, id_to_vector, name_to_id, id_to_name, labels,
    restaurant_name_to_id, order_repo, product_repo, alpha=ALPHA
)
content_engine = ContentEngine(product_repo, order_repo, collections["categories"])
fallback_engine = FallbackEngine(ratings_df)
orchestrator = RecommendationOrchestrator(
    collab_engine, content_engine, fallback_engine, id_to_name, id_to_vector, labels,
    restaurant_name_to_id, collections
)

test_user_ids = ["Alan Rifkin", "Maymouna Ndiaye"]

for user_id in test_user_ids:
    result = orchestrator.get_recommendations(user_id, top_n=5)
    if result:
        pprint(result, indent=2)
        save_recommendations_to_mongo(collections["db"], user_id, result)
    else:
        print(f"[ERROR] Failed to generate recommendations for user {user_id}")
