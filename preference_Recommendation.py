from pymongo import MongoClient
import pandas as pd

# Mongo connection
client = MongoClient("mongodb://localhost:27017")
db = client["recommender_db"]
restaurants_col = db["Restaurents"]
user_preferences_col = db["user_preferences"]

def get_top_restaurants_by_category(category_name, top_n=3):
    pipeline = [
        {"$match": {"maincuisineType": category_name}},
        {"$sort": {"averageRating": -1}},
        {"$limit": top_n},
        {"$project": {"_id": 1, "nom": 1, "averageRating": 1}}
    ]
    top_restaurants = list(restaurants_col.aggregate(pipeline))
    
    if not top_restaurants:
        fallback_pipeline = [
            {"$sort": {"averageRating": -1}},
            {"$limit": top_n},
            {"$project": {"_id": 1, "nom": 1, "averageRating": 1}}
        ]
        top_restaurants = list(restaurants_col.aggregate(fallback_pipeline))

    return [(r["nom"], str(r["_id"]), r["averageRating"]) for r in top_restaurants if r.get("averageRating") is not None]

def generate_first_time_recommendations(user_id):
    user_prefs = user_preferences_col.find_one({"userId": user_id})
    if not user_prefs or "categoryNames" not in user_prefs:
        print(f"[WARNING] No preferences found for user {user_id}")
        return {
            "user_id": user_id,
            "recommendations": [],
            "message": f"No preferences found for user {user_id}"
        }

    preferred_categories = user_prefs["categoryNames"]
    recommendations = {}

    for category in preferred_categories:
        top_restaurants = get_top_restaurants_by_category(category, top_n=3)
        if top_restaurants:
            recommendations[category] = top_restaurants
        else:
            print(f"[WARNING] No valid restaurants found for category: {category}")

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "message": f"Recommendations generated for user {user_id} based on preferences"
    }
