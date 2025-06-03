import numpy as np
from mmr import apply_mmr

class RecommendationOrchestrator:
    def __init__(self, collaborative_engine, content_engine, fallback_engine, id_to_name, id_to_vector, labels, restaurant_name_to_id, collections, lambda_mmr=0.7):
        self.collab = collaborative_engine
        self.content = content_engine
        self.fallback = fallback_engine
        self.id_to_name = id_to_name
        self.id_to_vector = id_to_vector
        self.labels = labels
        self.restaurant_name_to_id = restaurant_name_to_id
        self.collections = collections

        # Directly use lambda_mmr passed in (no Mongo fetch)
        self.lambda_mmr = lambda_mmr

    def get_recommendations(self, user_id, top_n=5):
        top_restaurants, sims = self.collab.recommend_restaurants(user_id, top_n)
        if top_restaurants is not None:
            print(f"[DEBUG] Using collaborative recommendations for {user_id}")
            return self.format_recommendation_result(user_id, top_restaurants, sims)

        fallback_recs = self.fallback.preference_fallback(user_id, top_n)
        if fallback_recs is not None:
            print(f"[DEBUG] Using preference fallback for {user_id}")
            return {
                "Recommendations": fallback_recs,
                "Products": {},
                "Restaurants Rated by Target User": [],
                "Neighboring Users (IDs)": [],
                "Neighboring Users (Names)": [],
                "Restaurants Rated by Neighbors": {},
                "Neighbor Preferences": {},
                "Target User Preference": {},
                "Message": f"Fallback recommendations based on preferences"
            }

        global_recs = self.fallback.global_popular_restaurants(top_n)
        if global_recs is not None:
            print(f"[DEBUG] Using global fallback for {user_id}")
            return {
                "Recommendations": [(r, self.restaurant_name_to_id.get(r.strip().lower(), "Unknown ID")) for r in global_recs["Restaurant"].tolist()],
                "Products": {},
                "Restaurants Rated by Target User": [],
                "Neighboring Users (IDs)": [],
                "Neighboring Users (Names)": [],
                "Restaurants Rated by Neighbors": {},
                "Neighbor Preferences": {},
                "Target User Preference": {},
                "Message": f"Global fallback recommendations"
            }

        print(f"[ERROR] No recommendations available for {user_id}")
        return None

    def format_recommendation_result(self, user_id, top_restaurants, sims):
        products_col = self.collections["products"]
        restaurants_col = self.collections["restaurants"]
        recommended_products = {}

        # Prepare MMR on restaurants with safe lookup
        top_restaurant_names = top_restaurants["Restaurant"].tolist()
        relevance_scores = top_restaurants["Weighted Rating"].tolist()
        categories = []
        for r in top_restaurant_names:
            rest_doc = restaurants_col.find_one({"nom": {"$regex": f"^{r}$", "$options": "i"}})
            if rest_doc:
                categories.append(rest_doc.get("internationalCuisine", ""))
            else:
                print(f"[WARNING] Restaurant '{r}' not found in DB — assigning empty category.")
                categories.append("")

        mmr_restaurants = apply_mmr(top_restaurant_names, relevance_scores, categories, self.lambda_mmr, len(top_restaurant_names))

        for rest_name in mmr_restaurants:
            rest_id = self.restaurant_name_to_id.get(rest_name.strip().lower())

            if not rest_id:
                recommended_products[rest_name] = "No restaurant ID found"
                continue

            user_has_history = self.collab.order_repository.user_has_history_with_restaurant(user_id, rest_id)

            if user_has_history:
                products = self.content.recommend_products_for_user(client_id=user_id, restaurant_id=rest_id, top_n=5)
                if not products:
                    fallback_products = self.collab.product_repository.get_products_by_restaurant(rest_id, limit=5)
                    products = [self.format_product(p) for p in fallback_products] if fallback_products else "No products available"
            else:
                fallback_products = self.collab.product_repository.get_products_by_restaurant(rest_id, limit=5)
                products = [self.format_product(p) for p in fallback_products] if fallback_products else "No products available"

            # Apply MMR on products if available
            if isinstance(products, list):
                product_names = [p["name"] for p in products]
                relevance_scores = [1.0 for _ in products]  # assuming equal relevance
                categories = [p.get("categorieName", "") for p in products]

                mmr_products = apply_mmr(product_names, relevance_scores, categories, self.lambda_mmr, min(5, len(products)))
                products = [p for p in products if p["name"] in mmr_products]

            recommended_products[rest_name] = products

        neighbor_names = [self.id_to_name.get(uid, f"User_{uid}") for uid, _ in sims]
        neighbor_info = {
            self.id_to_name.get(uid, uid): [r for r in top_restaurants["Restaurant"].tolist()]
            for uid, _ in sims
        }
        neighbor_prefs = {
            self.id_to_name.get(uid, uid): dict(zip(self.labels, self.id_to_vector[uid].round(4).tolist()))
            for uid, _ in sims
        }
        target_vec = self.id_to_vector.get(user_id)
        target_pref = dict(zip(self.labels, target_vec.round(4).tolist())) if isinstance(target_vec, np.ndarray) else {}

        return {
            "Recommendations": [(r, self.restaurant_name_to_id.get(r.strip().lower(), "Unknown ID")) for r in mmr_restaurants],
            "Products": recommended_products,
            "Restaurants Rated by Target User": [],
            "Neighboring Users (IDs)": [uid for uid, _ in sims],
            "Neighboring Users (Names)": neighbor_names,
            "Restaurants Rated by Neighbors": neighbor_info,
            "Neighbor Preferences": neighbor_prefs,
            "Target User Preference": target_pref,
            "Message": f"Recommendations generated with MMR diversification"
        }

    def format_product(self, product):
        return {
            "name": product.get("name", ""),
            "price": product.get("price", ""),
            "description": product.get("description", ""),
            "categorieName": product.get("categorieName", "")
        }
