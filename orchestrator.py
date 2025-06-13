import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json

from mmr import apply_mmr


class RecommendationOrchestrator:
    """
    Coordinates collaborative, content-based, and fallback engines.
    *NO* Redis caching: every call runs fresh.
    """

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

        # handy shorthands
        self.products_col = self.collections["products"]
        self.restaurants_col = self.collections["restaurants"]
        self.feedback_col = self.collections["RestaurentReaction"]

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def get_recommendations(self, user_id: str, top_n: int = 5):
        """
        Main entrypoint used by api.py.  
        Returns a dict (or None) WITHOUT touching Redis.
        """
        top_restaurants, sims = self.collab.recommend_restaurants(user_id, top_n)

        if top_restaurants is not None:
            return self._format_recommendation_result(
                user_id, top_restaurants, sims
            )

        # ----- Fallback 1: preference-based ----------------------------------
        fallback_recs = self.fallback.preference_fallback(user_id, top_n)
        if fallback_recs is not None:
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

        # ----- Fallback 2: global popular ------------------------------------
        global_recs = self.fallback.global_popular_restaurants(top_n)
        if global_recs is not None:
            return {
                "Recommendations": [
                    (
                        r,
                        self.restaurant_name_to_id.get(r.strip().lower(), "Unknown"),
                    )
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

        return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _format_recommendation_result(self, user_id, top_restaurants, sims):
        """
        MM-Reranks restaurants, grafts on product suggestions, and produces
        the large JSON payload expected by the API layer.
        """
        recommended_products = {}

        # ---------- Phase 1: Prep names / scores ---------------------------
        top_names = top_restaurants["Restaurant"].tolist()[:15]
        relevance = top_restaurants["Weighted Rating"].tolist()[:15]
        norm_names = [n.strip().lower() for n in top_names]

        # ---------- Phase 2: Bulk-fetch Mongo docs in parallel -------------
        with ThreadPoolExecutor(max_workers=3) as exe:
            feedback_f = exe.submit(
                lambda: list(self.feedback_col.find({"userId": user_id}))
            )
            rest_docs_f = exe.submit(
                lambda: list(
                    self.restaurants_col.find({"normalizedName": {"$in": norm_names}})
                )
            )
            rest_ids = [
                self.restaurant_name_to_id.get(n) for n in norm_names if self.restaurant_name_to_id.get(n)
            ]
            prod_docs_f = exe.submit(
                lambda: list(
                    self.products_col.find({"restaurantId": {"$in": rest_ids}})
                )
            )

            feedback_docs = feedback_f.result()
            rest_docs = rest_docs_f.result()
            product_docs = prod_docs_f.result()

        # ---------- Phase 3: Feedback boosts -------------------------------
        user_likes    = {d["restaurantId"] for d in feedback_docs if d["reaction"] == "LIKE"}
        user_dislikes = {d["restaurantId"] for d in feedback_docs if d["reaction"] == "DISLIKE"}

        name_to_cat = {
            d["normalizedName"]: d.get("internationalCuisine", "")
            for d in rest_docs
        }
        categories = [name_to_cat.get(n, "") for n in norm_names]
        mmr_names  = apply_mmr(top_names, relevance, categories, self.lambda_mmr, len(top_names))

        rest_scores = {}
        for name in mmr_names:
            rid = self.restaurant_name_to_id.get(name.strip().lower())
            if not rid:
                continue
            boost = self.like_weight if rid in user_likes else 0
            boost -= self.dislike_penalty if rid in user_dislikes else 0
            rest_scores[rid] = 1.0 + boost

        sorted_rids = [rid for rid, _ in sorted(rest_scores.items(), key=lambda x: x[1], reverse=True)]

        # ---------- Phase 4: Prepare product map ---------------------------
        prod_map = {}
        for d in product_docs:
            rid = d["restaurantId"]
            prod_map.setdefault(rid, []).append(self._format_product(d))

        # ---------- Phase 5: Build restaurant -> product suggestions -------
        for rid in sorted_rids:
            rest_name = next(
                (n for n, _id in self.restaurant_name_to_id.items() if _id == rid),
                None,
            )
            if not rest_name:
                continue

            has_hist = self.collab.order_repository.user_has_history_with_restaurant(
                user_id, rid
            )
            if has_hist:
                products = self.content.recommend_products_for_user(
                    user_id, rid, top_n=5
                )
            else:
                products = prod_map.get(rid, [])

            if isinstance(products, list) and len(products) > 5:
                names = [p["name"] for p in products]
                cats  = [p.get("categorieName", "") for p in products]
                keep  = apply_mmr(names, [1.0]*len(names), cats, self.lambda_mmr, 5)
                products = [p for p in products if p["name"] in keep]

            recommended_products[rest_name] = products or "No products available"

        # ---------- Phase 6: Neighbors & prefs -----------------------------
        neighbor_names = [self.id_to_name.get(uid, f"User_{uid}") for uid, _ in sims]
        neighbor_info  = {
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
            "Message": "Optimized for first-time and repeat calls",
        }

    # ------------------------------------------------------------------ #
    # Static helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_product(doc):
        return {
            "name":          doc.get("name", ""),
            "price":         doc.get("price", ""),
            "description":   doc.get("description", ""),
            "categorieName": doc.get("categorieName", ""),
        }
