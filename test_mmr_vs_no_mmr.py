import yaml
import pandas as pd
import pickle
from pprint import pprint
import csv

from data_loader import get_mongo_client, get_collections, reload_users
from model_handler import BERTModelHandler
from collaborative_engine import CollaborativeEngine
from content_engine import ContentEngine
from fallback_engine import FallbackEngine
from orchestrator import RecommendationOrchestrator
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
LAMBDA_MMR = config.get('lambda_mmr', 0.7)

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

# === Load external users file ===
user_csv = pd.read_csv("C:/Users/21655/Downloads/recommender_db.Users.csv")
user_list = user_csv[['User', '_id']].head(10).to_dict(orient='records')  # test top 10 users

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

# === Engines setup
collab_engine = CollaborativeEngine(
    model_handler, collections, id_to_vector, name_to_id, id_to_name, labels,
    restaurant_name_to_id, order_repo, product_repo, alpha=ALPHA
)
content_engine = ContentEngine(product_repo, order_repo, collections["categories"])
fallback_engine = FallbackEngine(ratings_df)

# === Prepare CSV output
output_file = 'mmr_test_results.csv'
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['User ID', 'Username', 'Mode', 'Top Recommendations'])

    for user in user_list:
        user_id = user['_id']
        username = user['User']

        # No MMR run
        collections["config"] = {"lambda_mmr": 1.0}
        orchestrator_no_mmr = RecommendationOrchestrator(
            collab_engine, content_engine, fallback_engine, id_to_name, id_to_vector, labels,
            restaurant_name_to_id, collections
        )
        result_no_mmr = orchestrator_no_mmr.get_recommendations(user_id, top_n=5)
        if result_no_mmr:
            recs_no_mmr = [r[0] for r in result_no_mmr["Recommendations"]]
            writer.writerow([user_id, username, 'no_mmr', recs_no_mmr])

        # With MMR run
        collections["config"]["lambda_mmr"] = LAMBDA_MMR
        orchestrator_with_mmr = RecommendationOrchestrator(
            collab_engine, content_engine, fallback_engine, id_to_name, id_to_vector, labels,
            restaurant_name_to_id, collections
        )
        result_with_mmr = orchestrator_with_mmr.get_recommendations(user_id, top_n=5)
        if result_with_mmr:
            recs_with_mmr = [r[0] for r in result_with_mmr["Recommendations"]]
            writer.writerow([user_id, username, 'with_mmr', recs_with_mmr])

print(f"✅ MMR test results saved to {output_file}")
