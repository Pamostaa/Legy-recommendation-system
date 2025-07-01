import logging
import pandas as pd
from preference_Recommendation import generate_first_time_recommendations

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class FallbackEngine:
    def __init__(self, ratings_df, collections):
        self.ratings = ratings_df.copy() if ratings_df is not None else pd.DataFrame()
        self.collections = collections
        self.restaurants_col = collections.get("restaurants")
        logging.debug(f"Ratings DataFrame shape: {self.ratings.shape}")
        if self.restaurants_col is None:
            logging.warning("restaurants_col not found in collections. Global popular recommendations may fail.")

    def preference_fallback(self, user_id, top_n=5):
        logging.debug(f"Calling preference_fallback for user {user_id}")
        try:
            pref_result = generate_first_time_recommendations(user_id)
            logging.debug(f"Preference result: {pref_result}")
            if pref_result and pref_result.get("recommendations"):
                flat_recs = [
                    (r[0], r[1]) 
                    for cat in pref_result["recommendations"].values() 
                    for r in cat
                ][:top_n]
                logging.debug(f"Flattened recommendations: {flat_recs}")
                return flat_recs
            else:
                logging.warning(f"No preference-based recommendations for user {user_id}")
                return None
        except Exception as e:
            logging.error(f"Error in preference_fallback for user {user_id}: {str(e)}")
            return None

    def global_popular_restaurants(self, top_n=5):
        if self.ratings.empty and self.restaurants_col is None:
            logging.warning("Ratings DataFrame is empty and restaurants_col is unavailable — no global fallback available")
            return []

        if not self.ratings.empty:
            if 'score' not in self.ratings.columns:
                logging.warning("No 'score' column found in ratings DataFrame.")
            else:
                self.ratings['score'] = self.ratings['score'].astype(str).str.replace(',', '.')
                self.ratings['score'] = pd.to_numeric(self.ratings['score'], errors='coerce')
                popular = self.ratings.groupby("Restaurant")["score"].mean().reset_index()
                popular = popular.sort_values(by="score", ascending=False).head(top_n)
                if 'restaurantId' in self.ratings.columns:
                    popular = popular.merge(self.ratings[["Restaurant", "restaurantId"]].drop_duplicates(), on="Restaurant", how="left")
                    return [(row["Restaurant"], str(row["restaurantId"]) if pd.notna(row["restaurantId"]) else "Unknown", row["score"]) for _, row in popular.iterrows()]
                return [(row["Restaurant"], "Unknown", row["score"]) for _, row in popular.iterrows()]

        if self.restaurants_col is not None:
            try:
                cursor = self.restaurants_col.find({}, {"_id": 1, "nom": 1, "averageRating": 1})
                df = pd.DataFrame(list(cursor))
                if not df.empty and 'averageRating' in df.columns:
                    df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
                    top_df = df.sort_values(by="averageRating", ascending=False).head(top_n).dropna(subset=['averageRating'])
                    return [
                        (row['nom'], str(row['_id']), row['averageRating']) 
                        for _, row in top_df.iterrows()
                    ]
                else:
                    logging.warning("No valid 'averageRating' data in restaurants collection.")
            except Exception as e:
                logging.error(f"Error querying restaurants collection: {str(e)}")
        return []

    def __repr__(self):
        return f"FallbackEngine(ratings_shape={self.ratings.shape}, collections_available={bool(self.collections)})"