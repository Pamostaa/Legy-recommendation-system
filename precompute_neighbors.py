import yaml
from concurrent.futures import ThreadPoolExecutor
from model_handler import BERTModelHandler
from collaborative_engine import CollaborativeEngine
from data_loader import get_mongo_client, get_collections, reload_users
from product_repository import ProductRepository
from order_repository import OrderRepository
from drive_utils import ensure_model_files

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

model_path = config["model_path"]
vectors_path = config["vectors_path"]
alpha = config.get("alpha", 0.01)

# === Ensure model files are available ===
ensure_model_files()

# === Initialize model handler ===
model_handler = BERTModelHandler(model_path, vectors_path)
id_to_vector = model_handler.id_to_vector

# === Mongo connection ===
client = get_mongo_client()
collections = get_collections(client)
db = client[collections["users"].database.name]  # Ensure correct DB name
neighbors_collection = db["user_neighbors"]

# === Load data ===
users_df, name_to_id, id_to_name = reload_users(collections["users"])
labels = ["cares_about_food_quality", "cares_about_service_speed", "cares_about_price", "cares_about_cleanliness"]

restaurant_name_to_id = {
    r.get("nom").strip().lower(): str(r["_id"])
    for r in collections["restaurants"].find()
    if "nom" in r and "_id" in r
}

# === Init engine ===
product_repo = ProductRepository(collections["products"])
order_repo = OrderRepository(collections["orders"])

engine = CollaborativeEngine(
    model_handler, collections, id_to_vector, name_to_id, id_to_name,
    labels, restaurant_name_to_id, order_repo, product_repo, alpha
)

# === Processing function ===
def process_user(user_id):
    try:
        if neighbors_collection.find_one({"user_id": user_id}):
            print(f"⏭️ Skipping already processed user {user_id}")
            return

        neighbors = engine.get_top_neighbors(user_id, top_n=10)
        if not neighbors:
            print(f"⚠️ No neighbors for user {user_id}")
            return

        doc = {
            "user_id": user_id,
            "neighbors": [{"user_id": n_id, "score": float(sim)} for n_id, sim in neighbors]
        }
        neighbors_collection.insert_one(doc)
        print(f"✅ Saved neighbors for user {user_id}")

    except Exception as e:
        print(f"❌ Error with user {user_id}: {e}")

# === Run in parallel ===
if __name__ == "__main__":
    all_user_ids = list(id_to_name.keys())
    print(f"[INFO] Computing neighbors for {len(all_user_ids)} users...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_user, all_user_ids)

    print("[DONE] Precomputation complete.")
