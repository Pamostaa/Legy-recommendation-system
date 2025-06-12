import numpy as np
from mmr import apply_mmr
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

class RecommendationOrchestrator:
    def __init__(self, collaborative_engine, content_engine, fallback_engine,
                 id_to_name, id_to_vector, labels, restaurant_name_to_id, collections,
                 lambda_mmr=0.7, like_weight=0.1, dislike_penalty=0.2,
                 redis_client=None, cache_expiry=864000):
        self.collab = collaborative_engine
        self.content = content_engine
        self.fallback = fallback_engine
        self.id_to_name = id_to_name
        self.id_to_vector = id_to_vector
        self.labels = labels
        self.restaurant_name_to_id = restaurant_name_to_id
        self.collections = collections
        self.lambda_mmr = lambda_mmr
        self.like_weight = like_weight
        self.dislike_penalty = dislike_penalty
        self.redis = redis_client
        self.cache_expiry = cache_expiry

    def get_recommendations(self, user_id, top_n=5):
        cache_key = f"recommendations:full:{user_id}"
        if self.redis:
            cached = self.redis.get(cache_key)
            if cached:
                print(f"[CACHE HIT] Full recommendation for {user_id}")
                return json.loads(cached)

        collab_key = f"collab:{user_id}"
        top_restaurants, sims = None, []

        if self.redis:
            cached_collab = self.redis.get(collab_key)
            if cached_collab:
                cached_obj = json.loads(cached_collab)
                top_restaurants = pd.DataFrame(cached_obj["top_restaurants"])
                sims = cached_obj["sims"]
                print(f"[CACHE HIT] Collaborative recs for {user_id}")
            else:
                top_restaurants, sims = self.collab.recommend_restaurants(user_id, top_n)
                if top_restaurants is not None:
                    collab_payload = {
                        "top_restaurants": top_restaurants.to_dict(orient="records"),
                        "sims": sims
                    }
                    self.redis.setex(collab_key, self.cache_expiry, json.dumps(collab_payload))
        else:
            top_restaurants, sims = self.collab.recommend_restaurants(user_id, top_n)

        if top_restaurants is not None:
            result = self.format_recommendation_result(user_id, top_restaurants, sims)
            if self.redis:
                self.redis.setex(cache_key, self.cache_expiry, json.dumps(result))
            return result

        # fallback 1
        fallback_recs = self.fallback.preference_fallback(user_id, top_n)
        if fallback_recs is not None:
            result = {
                "Recommendations": fallback_recs,
                "Products": {},
                "Restaurants Rated by Target User": [],
                "Neighboring Users (IDs)": [],
                "Neighboring Users (Names)": [],
                "Restaurants Rated by Neighbors": {},
                "Neighbor Preferences": {},
                "Target User Preference": {},
                "Message": "Fallback recommendations based on preferences"
            }
            if self.redis:
                self.redis.setex(cache_key, self.cache_expiry, json.dumps(result))
            return result

        # fallback 2
        global_recs = self.fallback.global_popular_restaurants(top_n)
        if global_recs is not None:
            result = {
                "Recommendations": [(r, self.restaurant_name_to_id.get(r.strip().lower(), "Unknown ID"))
                                    for r in global_recs["Restaurant"].tolist()],
                "Products": {},
                "Restaurants Rated by Target User": [],
                "Neighboring Users (IDs)": [],
                "Neighboring Users (Names)": [],
                "Restaurants Rated by Neighbors": {},
                "Neighbor Preferences": {},
                "Target User Preference": {},
                "Message": "Global fallback recommendations"
            }
            if self.redis:
                self.redis.setex(cache_key, self.cache_expiry, json.dumps(result))
            return result

        return None

    def format_recommendation_result(self, user_id, top_restaurants, sims):
        products_col = self.collections["products"]
        restaurants_col = self.collections["restaurants"]
        feedback_col = self.collections["RestaurentReaction"]

        recommended_products = {}

        top_restaurant_names = top_restaurants["Restaurant"].tolist()[:15]
        relevance_scores = top_restaurants["Weighted Rating"].tolist()[:15]
        normalized_names = [r.strip().lower() for r in top_restaurant_names]

        with ThreadPoolExecutor(max_workers=3) as executor:
            feedback_future = executor.submit(lambda: list(feedback_col.find({"userId": user_id})))
            rest_docs_future = executor.submit(lambda: list(restaurants_col.find({"normalizedName": {"$in": normalized_names}})))
            restaurant_ids = [self.restaurant_name_to_id.get(r.lower()) for r in normalized_names if self.restaurant_name_to_id.get(r.lower())]
            product_docs_future = executor.submit(lambda: list(products_col.find({"restaurantId": {"$in": restaurant_ids}})))

            feedback_docs = feedback_future.result()
            rest_docs = rest_docs_future.result()
            product_docs = product_docs_future.result()

        user_likes = {doc["restaurantId"] for doc in feedback_docs if doc["reaction"] == "LIKE"}
        user_dislikes = {doc["restaurantId"] for doc in feedback_docs if doc["reaction"] == "DISLIKE"}

        name_to_category = {doc["normalizedName"]: doc.get("internationalCuisine", "") for doc in rest_docs}
        categories = [name_to_category.get(r.strip().lower(), "") for r in top_restaurant_names]
        mmr_restaurants = apply_mmr(top_restaurant_names, relevance_scores, categories, self.lambda_mmr, len(top_restaurant_names))

        restaurant_scores = {}
        for rest_name in mmr_restaurants:
            rest_id = self.restaurant_name_to_id.get(rest_name.strip().lower())
            if not rest_id:
                continue
            feedback_boost = self.like_weight if rest_id in user_likes else 0
            feedback_boost -= self.dislike_penalty if rest_id in user_dislikes else 0
            restaurant_scores[rest_id] = 1.0 + feedback_boost

        sorted_restaurants = sorted(restaurant_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_rest_ids = [rest_id for rest_id, _ in sorted_restaurants]

        product_map = {}
        for doc in product_docs:
            rid = doc["restaurantId"]
            product_map.setdefault(rid, []).append(self.format_product(doc))

        for rest_id in sorted_rest_ids:
            rest_name = next((name for name, id_ in self.restaurant_name_to_id.items() if id_ == rest_id), None)
            if not rest_name:
                continue

            user_has_history = self.collab.order_repository.user_has_history_with_restaurant(user_id, rest_id)
            products = self.content.recommend_products_for_user(user_id, rest_id, top_n=5) if user_has_history else product_map.get(rest_id, [])

            if isinstance(products, list) and len(products) > 5:
                names = [p["name"] for p in products]
                scores = [1.0 for _ in products]
                cats = [p.get("categorieName", "") for p in products]
                mmr_products = apply_mmr(names, scores, cats, self.lambda_mmr, 5)
                products = [p for p in products if p["name"] in mmr_products]

            recommended_products[rest_name] = products or "No products available"

        neighbor_names = [self.id_to_name.get(uid, f"User_{uid}") for uid, _ in sims]
        neighbor_info = {
            self.id_to_name.get(uid, uid): top_restaurants["Restaurant"].tolist()
            for uid, _ in sims
        }
        neighbor_prefs = {
            self.id_to_name.get(uid, uid): dict(zip(self.labels, self.id_to_vector[uid].round(4).tolist()))
            for uid, _ in sims
        }
        target_vec = self.id_to_vector.get(user_id)
        target_pref = dict(zip(self.labels, target_vec.round(4).tolist())) if isinstance(target_vec, np.ndarray) else {}

        return {
            "Recommendations": [(name, id_) for name, id_ in self.restaurant_name_to_id.items() if id_ in dict(sorted_restaurants)],
            "Products": recommended_products,
            "Restaurants Rated by Target User": [],
            "Neighboring Users (IDs)": [uid for uid, _ in sims],
            "Neighboring Users (Names)": neighbor_names,
            "Restaurants Rated by Neighbors": neighbor_info,
            "Neighbor Preferences": neighbor_prefs,
            "Target User Preference": target_pref,
            "Message": "Optimized for first-time and repeat calls"
        }

    def format_product(self, product):
        return {
            "name": product.get("name", ""),
            "price": product.get("price", ""),
            "description": product.get("description", ""),
            "categorieName": product.get("categorieName", "")
        }
