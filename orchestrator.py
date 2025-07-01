import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
import traceback
from bson import ObjectId
from typing import List, Dict, Any

from mmr import apply_mmr

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

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
        self.feedback_col = self.collections["restaurant_reactions"]  # Fixed: Use "restaurant_reactions"
        logging.debug("Initialized RecommendationOrchestrator")
        logging.debug(f"Available collections: {list(self.collections.keys())}")
        logging.debug(f"Available user IDs: {list(self.id_to_name.keys())[:5]}")
        logging.debug(f"Available restaurant IDs: {list(self.restaurant_name_to_id.values())[:5]}")

    def get_recommendations(self, user_id: str, top_n: int = 5) -> Dict[str, Any]:
        logging.debug(f"Fetching recommendations for user {user_id}")
        
        try:
            top_restaurants, sims = self.collab.recommend_restaurants(user_id, top_n)
            if top_restaurants is not None and not top_restaurants.empty:
                logging.debug(f"Collaborative recommendations: {top_restaurants.to_dict()}")
                return self._format_recommendation_result(user_id, top_restaurants, sims, top_n)

            # Fallback 1: preference-based
            fallback_recs = self.fallback.preference_fallback(user_id, top_n)
            logging.debug(f"Preference fallback result: {fallback_recs}")
            if fallback_recs:
                logging.debug(f"Using preference-based fallback for {user_id}")
                return {
                    "Recommendations": fallback_recs[:top_n],
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
            logging.debug(f"Global popular fallback result: {global_recs}")
            if global_recs:
                recommendations = [
                    (r[0], r[1]) for r in global_recs[:top_n]
                ]
                logging.debug(f"Using global popular fallback for {user_id}: {recommendations}")
                return {
                    "Recommendations": recommendations,
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
            return {
                "Recommendations": [],
                "Products": {},
                "Message": "No recommendations available due to insufficient user data"
            }
        except Exception as e:
            logging.error(f"Error in get_recommendations for user {user_id}: {str(e)}\n{traceback.format_exc()}")
            raise

    def _format_recommendation_result(self, user_id: str, top_restaurants: pd.DataFrame, sims: List, top_n: int) -> Dict[str, Any]:
        recommended_products = {}
        rest_ids = top_restaurants["RestaurantId"].tolist()[:top_n * 3]
        logging.debug(f"Initial restaurant IDs: {rest_ids}")
        rest_ids = [rid for rid in rest_ids if isinstance(rid, str) and ObjectId.is_valid(rid)]
        logging.debug(f"Valid restaurant IDs: {rest_ids}")
        query = {"_id": {"$in": [ObjectId(rid) for rid in rest_ids]}}
        logging.debug(f"Query for restaurants: {query}")
        rest_docs = list(self.restaurants_col.find(query))
        logging.debug(f"Found {len(rest_docs)} restaurant documents with query: {query}")
        id_to_name = {str(d["_id"]): d.get("nom", "").strip().lower() for d in rest_docs}
        top_names = [id_to_name.get(rid, "Unknown") for rid in rest_ids]
        relevance = top_restaurants["WeightedRating"].tolist()[:top_n * 3]
        norm_names = [n for n in top_names if n != "Unknown"]
        logging.debug(f"Normalized restaurant names: {norm_names}")

        if len(norm_names) < top_n:
            logging.warning(f"Only {len(norm_names)} valid restaurants found for user {user_id}, falling back")
            global_recs = self.fallback.global_popular_restaurants(top_n)
            if global_recs:
                additional_names = [r[0].strip().lower() for r in global_recs if r[0].strip().lower() not in norm_names]
                norm_names.extend(additional_names[:top_n - len(norm_names)])
                relevance.extend([1.0] * (top_n - len(relevance)))
                logging.debug(f"Added fallback restaurants: {norm_names}")

        with ThreadPoolExecutor(max_workers=3) as exe:
            feedback_f = exe.submit(lambda: list(self.feedback_col.find({"userId": user_id})))
            rest_docs_f = exe.submit(lambda: rest_docs)
            prod_docs_f = exe.submit(lambda: list(self.products_col.find({"restaurantId": {"$in": rest_ids}})))
            feedback_docs = feedback_f.result()
            rest_docs = rest_docs_f.result()
            product_docs = prod_docs_f.result()

        logging.debug(f"Found {len(product_docs)} products for restaurants {rest_ids}")

        user_likes = {d["restaurantId"] for d in feedback_docs if d.get("reaction") == "LIKE"}
        user_dislikes = {d["restaurantId"] for d in feedback_docs if d.get("reaction") == "DISLIKE"}

        name_to_cat = {d["nom"].strip().lower(): d.get("mainCuisineType", "").lower() for d in rest_docs}
        categories = [name_to_cat.get(n, "") for n in norm_names]
        mmr_names = apply_mmr(norm_names, relevance[:len(norm_names)], categories, self.lambda_mmr, top_n)
        logging.debug(f"MMR selected restaurants: {mmr_names}")

        rest_scores = {}
        for name in mmr_names:
            rid = self.restaurant_name_to_id.get(name, None)
            if not rid:
                logging.warning(f"No restaurant ID for name: {name}")
                continue
            boost = self.like_weight if rid in user_likes else 0
            boost -= self.dislike_penalty if rid in user_dislikes else 0
            rest_scores[rid] = 1.0 + boost

        sorted_rids = [rid for rid, _ in sorted(rest_scores.items(), key=lambda x: x[1], reverse=True)][:top_n]
        logging.debug(f"Sorted restaurant IDs: {sorted_rids}")

        prod_map = {}
        for d in product_docs:
            rid = d["restaurantId"]
            prod_map.setdefault(rid, []).append(str(d["_id"]))

        logging.debug(f"Product map: {prod_map}")

        for rid in sorted_rids:
            rest_name = next((n for n, _id in self.restaurant_name_to_id.items() if _id == rid), None)
            if not rest_name:
                logging.warning(f"No restaurant name for ID: {rid}")
                continue

            products = self.content.cbf.recommend_for_user(user_id, rid, top_n=top_n)
            product_ids = [str(prod["_id"]) for prod in products if "_id" in prod]
            logging.debug(f"Initial products for {rest_name} (ID: {rid}): {product_ids}")

            if len(product_ids) < top_n:
                remaining = top_n - len(product_ids)
                logging.debug(f"Fetching {remaining} additional products for {rest_name}")
                fallback_products = self.content.cbf.recommend_for_user(user_id, None, top_n=remaining)
                fallback_product_ids = [str(prod["_id"]) for prod in fallback_products if "_id" in prod]
                fallback_product_ids = [pid for pid in fallback_product_ids if pid not in product_ids]
                product_ids.extend(fallback_product_ids[:remaining])
                logging.debug(f"Added {len(fallback_product_ids[:remaining])} fallback products for {rest_name}")

            if product_ids:
                recommended_products[rest_name] = product_ids[:top_n]
                logging.debug(f"Final products for {rest_name}: {recommended_products[rest_name]}")

        neighbor_names = [self.id_to_name.get(uid, f"User_{uid}") for uid, _ in sims]
        neighbor_info = {self.id_to_name.get(uid, uid): norm_names for uid, _ in sims}
        neighbor_prefs = {
            self.id_to_name.get(uid, uid): dict(zip(self.labels, self.id_to_vector[uid].round(4).tolist()))
            for uid, _ in sims
        }
        target_vec = self.id_to_vector.get(user_id)
        target_pref = dict(zip(self.labels, target_vec.round(4).tolist())) if isinstance(target_vec, np.ndarray) else {}

        recommendations = [(n, self.restaurant_name_to_id.get(n, "Unknown")) for n in mmr_names[:top_n]]
        logging.debug(f"Final recommendations: {recommendations}")

        return {
            "Recommendations": recommendations,
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