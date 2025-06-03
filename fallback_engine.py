from preference_Recommendation import generate_first_time_recommendations

class FallbackEngine:
    def __init__(self, ratings_df):
        self.ratings = ratings_df

    def preference_fallback(self, user_id, top_n=5):
        pref_result = generate_first_time_recommendations(user_id)
        if pref_result.get("recommendations"):
            flat_recs = [(r[0], r[1]) for cat in pref_result["recommendations"].values() for r in cat][:top_n]
            return flat_recs
        return None

    def global_popular_restaurants(self, top_n=5):
        if self.ratings.empty:
            print("[WARNING] Ratings DataFrame is empty — no global fallback available")
            return None
        top_restaurants = self.ratings.groupby("Restaurant")["User Rating"].mean().reset_index().sort_values(by="User Rating", ascending=False)
        return top_restaurants.head(top_n)
