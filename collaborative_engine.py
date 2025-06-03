import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class CollaborativeEngine:
    def __init__(self, model_handler, collections, id_to_vector, name_to_id, id_to_name, labels, restaurant_name_to_id, order_repository, product_repository, alpha=0.01):
        self.model_handler = model_handler
        self.collections = collections
        self.reviews_col = collections["reviews"]
        self.id_to_vector = id_to_vector
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.labels = labels
        self.restaurant_name_to_id = restaurant_name_to_id
        self.order_repository = order_repository
        self.product_repository = product_repository
        self.alpha = alpha  # tunable decay parameter
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
            print(f"[DEBUG] No reviews found for {user_id} ({user_name})")
            return False

        vectors = [self.model_handler.compute_vector(r["Review"]) for r in user_reviews if r.get("Review")]
        if not vectors:
            print(f"[DEBUG] No review texts to vectorize for {user_id} ({user_name})")
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
            return 1.0  # no decay if missing
        days_since = (current_time - created_at).days
        return np.exp(-self.alpha * days_since)

    def recommend_restaurants(self, user_id, top_n=5):
        top_n = min(top_n, self.MAX_RECOMMENDATIONS)
        user_name = self.id_to_name.get(user_id, user_id)
        now = datetime.now()

        if user_id not in self.id_to_vector:
            print(f"[DEBUG] User {user_id} missing vector, rebuilding...")
            if not self.rebuild_user_profile(user_id):
                print(f"[WARNING] Failed to rebuild profile from reviews for {user_id}")
                return None, []

        user_rated = set(self.ratings[self.ratings["User"] == user_name]["Restaurant"])
        if not user_rated:
            print(f"[DEBUG] User {user_id} has no rated restaurants")
        count = len(user_rated)
        dynamic_neighbors = 11 if count > 10 else max(3, count + 2)

        sims = []
        for other_id in self.id_to_vector:
            if other_id == user_id:
                continue
            other_name = self.id_to_name.get(other_id, other_id)
            other_rated = set(self.ratings[self.ratings["User"] == other_name]["Restaurant"])
            overlap = len(user_rated & other_rated)
            if overlap > 0 and len(other_rated) <= dynamic_neighbors:
                new_items = len(other_rated - user_rated)
                score = overlap + new_items
                sims.append((other_id, score))

        sorted_similar = sorted(sims, key=lambda x: (-self.vector_similarity(user_id, x[0]), -x[1]))[:self.MAX_NEIGHBORS]
        neighbors = [u for u, _ in sorted_similar]

        if neighbors:
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
