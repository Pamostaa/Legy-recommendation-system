import json
import redis
import yaml
from concurrent.futures import ThreadPoolExecutor
from model_handler import BERTModelHandler
from collaborative_engine import CollaborativeEngine
from data_loader import get_mongo_client, get_collections, reload_users

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

model_path = config["model_path"]
vectors_path = config["vectors_path"]
alpha = config.get("alpha", 0.01)
redis_host = config.get("redis_host", "localhost")
redis_port = config.get("redis_port", 6379)
cache_expiry = 10 * 24 * 60 * 60  # 10 days

# === Redis client ===
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

# === Load Mongo and data ===
client = get_mongo_client()
collections = get_collections(client)

users_df, name_to_id, id_to_name = reload_users(collections["users"])
labels = ["cares_about_food_quality", "cares_about_service_speed", "cares_about_price", "cares_about_cleanliness"]

restaurant_name_to_id = {
    r.get("nom").strip().lower(): str(r["_id"])
    for r in collections["restaurants"].find()
    if "nom" in r and "_id" in r
}

# === Init model + engine ===
model_handler = BERTModelHandler(model_path, vectors_path)
id_to_vector = model_handler.id_to_vector
from product_repository import ProductRepository
from order_repository import OrderRepository

product_repo = ProductRepository(collections["products"])
order_repo = OrderRepository(collections["orders"])

engine = CollaborativeEngine(
    model_handler, collections, id_to_vector, name_to_id, id_to_name,
    labels, restaurant_name_to_id, order_repo, product_repo, alpha
)

# === Define processing function ===
def process_user(user_id):
    try:
        top_restaurants, sims = engine.recommend_restaurants(user_id, top_n=5)
        if top_restaurants is not None:
            payload = {
                "top_restaurants": top_restaurants.to_dict(orient="records"),
                "sims": sims
            }
            redis_client.setex(f"collab:{user_id}", cache_expiry, json.dumps(payload))
            print(f"✅ Cached for user {user_id}")
        else:
            print(f"⚠️ No recommendations for {user_id}")
    except Exception as e:
        print(f"❌ Error processing user {user_id}: {e}")

# === Run in threads ===
if __name__ == "__main__":
    all_user_ids = list(id_to_name.keys())  # or filter for active users
    print(f"[INFO] Precomputing neighbors for {len(all_user_ids)} users...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_user, all_user_ids)

    print("[DONE] Precomputation complete.")
