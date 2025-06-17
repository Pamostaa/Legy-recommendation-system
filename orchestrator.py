import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
from bson import ObjectId
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from mmr import apply_mmr

class RecommendationOrchestrator:
    def __init__(
        self,
        collaborative_engine,
        content_engine,
        fallback_engine,
        id_to_name,
        id_to_vector,
        labels,
        restaurant_name_to_id,
        collections,
        lambda_mmr=0.7,
        like_weight=0.1,
        dislike_penalty=0.2,
    ):
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
        self.products_col = self.collections["products"]
        self.restaurants_col = self.collections["restaurants"]
        self.feedback_col = self.collections["RestaurentReaction"]

    def get_recommendations(self, user_id: str, top_n: int = 5) -> Dict[str, Any]:
        logging.debug(f"Fetching recommendations for user {user_id}")
        
        top_restaurants, sims = self.collab.recommend_restaurants(user_id, top_n)
        if top_restaurants is not None:
            return self._format_recommendation_result(user_id, top_restaurants, sims, top_n)

        # Fallback 1: preference-based
        fallback_recs = self.fallback.preference_fallback(user_id, top_n)
        if fallback_recs is not None:
            logging.debug(f"Using preference-based fallback for {user_id}")
            return {
                "Recommendations": fallback_recs,
                "Products": {},
                "Restaurants Rated by Target User": [],
                "Neighboring Users (IDs)": [],
                "Neighboring Users (Names)": [],
                "Restaurants Rated by Neighbors": {},
                "Neighbor Preferences": {},
                "Target User Preference": {},
                "Message": "Fallback recommendations based on preferences",
            }

        # Fallback 2: global popular
        global_recs = self.fallback.global_popular_restaurants(top_n)
        if global_recs is not None:
            logging.debug(f"Using global popular fallback for {user_id}")
            return {
                "Recommendations": [
                    (r, self.restaurant_name_to_id.get(r.strip().lower(), "Unknown"))
                    for r in global_recs["Restaurant"].tolist()
                ],
                "Products": {},
                "Restaurants Rated by Target User": [],
                "Neighboring Users (IDs)": [],
                "Neighboring Users (Names)": [],
                "Restaurants Rated by Neighbors": {},
                "Neighbor Preferences": {},
                "Target User Preference": {},
                "Message": "Global fallback recommendations",
            }

        logging.warning(f"No recommendations for {user_id}")
        return {"error": "No recommendations available"}

    def _format_recommendation_result(self, user_id: str, top_restaurants: pd.DataFrame, sims: List, top_n: int) -> Dict[str, Any]:
        recommended_products = {}
        top_names = top_restaurants["Restaurant"].tolist()[:15]
        relevance = top_restaurants["Weighted Rating"].tolist()[:15]
        norm_names = [n.strip().lower() for n in top_names]

        with ThreadPoolExecutor(max_workers=3) as exe:
            feedback_f = exe.submit(lambda: list(self.feedback_col.find({"userId": user_id})))
            rest_docs_f = exe.submit(lambda: list(self.restaurants_col.find({"normalizedName": {"$in": norm_names}})))
            rest_ids = [self.restaurant_name_to_id.get(n) for n in norm_names if self.restaurant_name_to_id.get(n)]
            prod_docs_f = exe.submit(lambda: list(self.products_col.find({"restaurantId": {"$in": rest_ids}})))
            feedback_docs = feedback_f.result()
            rest_docs = rest_docs_f.result()
            product_docs = prod_docs_f.result()

        logging.debug(f"Found {len(product_docs)} products for restaurants {rest_ids}")

        user_likes = {d["restaurantId"] for d in feedback_docs if d["reaction"] == "LIKE"}
        user_dislikes = {d["restaurantId"] for d in feedback_docs if d["reaction"] == "DISLIKE"}

        name_to_cat = {d["normalizedName"]: d.get("internationalCuisine", "") for d in rest_docs}
        categories = [name_to_cat.get(n, "") for n in norm_names]
        mmr_names = apply_mmr(top_names, relevance, categories, self.lambda_mmr, len(top_names))

        rest_scores = {}
        for name in mmr_names:
            rid = self.restaurant_name_to_id.get(name.strip().lower())
            if not rid:
                continue
            boost = self.like_weight if rid in user_likes else 0
            boost -= self.dislike_penalty if rid in user_dislikes else 0
            rest_scores[rid] = 1.0 + boost

        sorted_rids = [rid for rid, _ in sorted(rest_scores.items(), key=lambda x: x[1], reverse=True)]

        prod_map = {}
        for d in product_docs:
            rid = d["restaurantId"]
            prod_map.setdefault(rid, []).append(str(d["_id"]))

        logging.debug(f"Product map: {prod_map}")

        products_added = 0
        for rid in sorted_rids:
            rest_name = next((n for n, _id in self.restaurant_name_to_id.items() if _id == rid), None)
            if not rest_name:
                continue

            products = self.content.cbf.recommend_for_user(user_id, rid, top_n=top_n)
            product_ids = [str(prod["_id"]) for prod in products if "_id" in prod]
            logging.debug(f"Products for {rest_name} (ID: {rid}): {product_ids}")

            if product_ids:
                recommended_products[rest_name] = product_ids
                products_added += len(product_ids)
                if products_added >= top_n:
                    break

        # Fallback: Similar products or random if no products found
        if products_added < top_n and self.collab.order_repository.user_has_history(user_id):
            remaining = top_n - products_added
            logging.debug(f"Fetching {remaining} similar products for {user_id}")
            similar_products = self.content.cbf.recommend_for_user(user_id, None, top_n=remaining)
            product_ids = [str(prod["_id"]) for prod in similar_products if "_id" in prod]
            if product_ids:
                recommended_products["Similar Products"] = product_ids
                products_added += len(product_ids)
                logging.debug(f"Added {len(product_ids)} similar products")

        # Final fallback: Random products for users without history or if still not enough
        if products_added < top_n:
            remaining = top_n - products_added
            logging.debug(f"Fetching {remaining} random products for {user_id}")
            random_products = self.content.cbf.recommend_for_user(user_id, None, top_n=remaining)
            product_ids = [str(prod["_id"]) for prod in random_products if "_id" in prod]
            if product_ids:
                recommended_products["Random Products"] = product_ids
                logging.debug(f"Added {len(product_ids)} random products")

        neighbor_names = [self.id_to_name.get(uid, f"User_{uid}") for uid, _ in sims]
        neighbor_info = {
            self.id_to_name.get(uid, uid): top_restaurants["Restaurant"].tolist()
            for uid, _ in sims
        }
        neighbor_prefs = {
            self.id_to_name.get(uid, uid): dict(
                zip(self.labels, self.id_to_vector[uid].round(4).tolist())
            )
            for uid, _ in sims
        }
        target_vec = self.id_to_vector.get(user_id)
        target_pref = (
            dict(zip(self.labels, target_vec.round(4).tolist()))
            if isinstance(target_vec, np.ndarray)
            else {}
        )

        return {
            "Recommendations": [
                (n, self.restaurant_name_to_id.get(n.strip().lower(), "unknown"))
                for n in mmr_names
            ],
            "Products": recommended_products,
            "Restaurants Rated by Target User": [],
            "Neighboring Users (IDs)": [uid for uid, _ in sims],
            "Neighboring Users (Names)": neighbor_names,
            "Restaurants Rated by Neighbors": neighbor_info,
            "Neighbor Preferences": neighbor_prefs,
            "Target User Preference": target_pref,
            "Message": "Optimized recommendations with product fallbacks",
        }

    @staticmethod
    def _format_product(doc):
        return {
            "name": doc.get("name", ""),
            "price": doc.get("price", ""),
            "description": doc.get("description", ""),
            "categorieName": doc.get("categorieName", ""),
        }