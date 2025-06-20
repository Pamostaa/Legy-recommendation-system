import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class CollaborativeEngine:
    def __init__(self, model_handler, collections, id_to_vector, name_to_id, id_to_name, labels,
                 restaurant_name_to_id, order_repository, product_repository, alpha=0.01):
        self.model_handler = model_handler
        self.collections = collections
        self.reviews_col = collections["reviews"]
        self.neighbors_col = collections["user_neighbors"]
        self.id_to_vector = id_to_vector
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.labels = labels
        self.restaurant_name_to_id = restaurant_name_to_id
        self.order_repository = order_repository
        self.product_repository = product_repository
        self.alpha = alpha
        self.MAX_NEIGHBORS = 10
        self.MAX_RECOMMENDATIONS = 10

        reviews = list(self.reviews_col.find({}))
        self.ratings = pd.DataFrame(reviews)
        if not self.ratings.empty:
            self.ratings.rename(columns={
                "User": "User",
                "restaurant_Id": "Restaurant",
                "User Rating": "User Rating",
                "createdAt": "createdAt"
            }, inplace=True)
            self.ratings['createdAt'] = pd.to_datetime(self.ratings['createdAt'], errors='coerce')

    def rebuild_user_profile(self, user_id):
        user_name = self.id_to_name.get(user_id, user_id)
        user_reviews = list(self.reviews_col.find({"User": user_name}))
        if not user_reviews:
            return False

        vectors = [self.model_handler.compute_vector(r["Review"]) for r in user_reviews if r.get("Review")]
        if not vectors:
            return False

        mean_vector = np.mean(vectors, axis=0)
        self.id_to_vector[user_id] = mean_vector
        self.model_handler.save_vectors()
        return True

    def vector_similarity(self, u1, u2):
        v1 = self.id_to_vector.get(u1)
        v2 = self.id_to_vector.get(u2)
        return cosine_similarity([v1], [v2])[0][0] if v1 is not None and v2 is not None else 0

    def apply_time_decay(self, created_at, current_time):
        if pd.isna(created_at):
            return 1.0
        days_since = (current_time - created_at).days
        return np.exp(-self.alpha * days_since)

    def get_precomputed_neighbors(self, user_id):
        doc = self.neighbors_col.find_one({"user_id": user_id})
        if not doc or "neighbors" not in doc:
            return []
        return [(n["user_id"], n["score"]) for n in doc["neighbors"]]

    def recommend_restaurants(self, user_id, top_n=5):
        top_n = min(top_n, self.MAX_RECOMMENDATIONS)
        user_name = self.id_to_name.get(user_id, user_id)
        now = datetime.now()

        if user_id not in self.id_to_vector:
            if not self.rebuild_user_profile(user_id):
                return None, []

        sorted_similar = self.get_precomputed_neighbors(user_id)
        if not sorted_similar:
            logging.warning(f"No precomputed neighbors for user {user_id}, computing dynamically")
            sorted_similar = self.get_top_neighbors(user_id, top_n=self.MAX_NEIGHBORS)
            if not sorted_similar:
                return None, []

        neighbors = [u for u, _ in sorted_similar]
        if not neighbors:
            return None, []

        user_rated = set(self.ratings[self.ratings["User"] == user_name]["Restaurant"])
        neighbor_names = [self.id_to_name.get(uid, uid) for uid in neighbors]
        neighbor_ratings = self.ratings[self.ratings["User"].isin(neighbor_names)].copy()
        neighbor_ratings['decay_factor'] = neighbor_ratings['createdAt'].apply(lambda x: self.apply_time_decay(x, now))
        neighbor_ratings['Weighted Rating'] = neighbor_ratings['User Rating'] * neighbor_ratings['decay_factor']

        recs = neighbor_ratings[~neighbor_ratings["Restaurant"].isin(user_rated)]
        top_restaurants = recs.groupby("Restaurant")["Weighted Rating"].mean().reset_index().sort_values(by="Weighted Rating", ascending=False)
        top_restaurants = top_restaurants.head(top_n)
        if not top_restaurants.empty:
            return top_restaurants, sorted_similar

        return None, []
    def get_top_neighbors(self, user_id, top_n=10):
        if user_id not in self.id_to_vector:
            success = self.rebuild_user_profile(user_id)
            if not success or user_id not in self.id_to_vector:
                return []

        user_vector = self.id_to_vector[user_id].reshape(1, -1)
        similarities = []
        vector_items = list(self.id_to_vector.items())
        for other_id, other_vector in vector_items:
            if other_id == user_id:
                continue
            score = cosine_similarity(user_vector, other_vector.reshape(1, -1))[0][0]
            similarities.append((other_id, score))

        top_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        return top_neighbors