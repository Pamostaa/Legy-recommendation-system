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
from user_repository import UserRepository
from product_repository import ProductRepository
from order_repository import OrderRepository

# === Load config ===
with open('config.yaml') as f:
    config = yaml.safe_load(f)

MODEL_PATH = config['model_path']
VECTORS_PATH = config['vectors_path']
ALPHA = config.get('alpha', 0.01)

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

# === Define weight sets to test ===
weight_sets = [
    (0.4, 0.4, 0.2, 0.0),  # no sentiment
    (0.3, 0.3, 0.2, 0.2),  # balanced
    (0.2, 0.2, 0.2, 0.4),  # sentiment heavy
    (0.5, 0.3, 0.1, 0.1),  # similarity heavy
]

test_user_id = "Alan Rifkin"
restaurant_id = "6811fbd550511129755f0af7"  # replace with a valid restaurantId

for weights in weight_sets:
    print(f"\n==== Testing weights: Ingredient={weights[0]}, Price={weights[1]}, Rating={weights[2]}, Sentiment={weights[3]} ====")

    # Initialize engines with this weight set
    model_handler = BERTModelHandler(MODEL_PATH, VECTORS_PATH)
    collab_engine = CollaborativeEngine(
        model_handler, collections, id_to_vector, name_to_id, id_to_name, labels,
        restaurant_name_to_id, order_repo, product_repo, alpha=ALPHA
    )
    content_engine = ContentEngine(product_repo, order_repo, weights=weights)
    fallback_engine = FallbackEngine(ratings_df)
    orchestrator = RecommendationOrchestrator(
        collab_engine, content_engine, fallback_engine, id_to_name, id_to_vector, labels,
        restaurant_name_to_id, collections
    )

    # Get recommendations
    result = orchestrator.get_recommendations(test_user_id, top_n=5)
    if result:
        pprint(result, indent=2)
    else:
        print("[ERROR] Failed to generate recommendations for this configuration.")
