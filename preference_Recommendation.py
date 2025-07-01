from pymongo import MongoClient
import pandas as pd
import os
import logging
from data_loader import get_mongo_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mongo connection - use environment variable or default to Docker container
client = get_mongo_client()
db = client["leggydb"]  # Updated to new database name
restaurants_col = db["restaurants"]  # Updated collection name to lowercase
user_preferences_col = db["user_preferences"]

def get_top_restaurants_by_category(category_name=None, top_n=3):
    """
    Retrieve top restaurants by category or across all categories if category_name is None.
    
    Args:
        category_name (str, optional): The cuisine type to filter restaurants. If None, returns top restaurants across all categories.
        top_n (int): Number of restaurants to return.
    
    Returns:
        list: List of tuples (restaurant_name, restaurant_id, average_rating).
    """
    pipeline = []
    if category_name:
        pipeline.append({"$match": {"mainCuisineType": category_name}})
    pipeline.extend([
        {"$sort": {"averageRating": -1}},
        {"$limit": top_n},
        {"$project": {"_id": 1, "nom": 1, "averageRating": 1}}
    ])
    top_restaurants = list(restaurants_col.aggregate(pipeline))
    
    # Log the results
    if not top_restaurants:
        logger.warning(f"No restaurants found for category: {category_name if category_name else 'all categories'}")
    
    return [(r["nom"], str(r["_id"]), r["averageRating"]) for r in top_restaurants if r.get("averageRating") is not None]

def generate_first_time_recommendations(user_id):
    """
    Generate restaurant recommendations for a user based on their preferences.
    If no preferences exist, return top restaurants across all categories.
    
    Args:
        user_id (str): The ID of the user to generate recommendations for.
    
    Returns:
        dict: Dictionary containing user_id, recommendations, and a message.
    """
    try:
        user_prefs = user_preferences_col.find_one({"userId": user_id})
        
        if not user_prefs or "categoryNames" not in user_prefs or not user_prefs["categoryNames"]:
            logger.warning(f"No preferences found for user {user_id}. Using fallback recommendations.")
            top_restaurants = get_top_restaurants_by_category(top_n=5)  # Get top 5 restaurants across all categories
            return {
                "user_id": user_id,
                "recommendations": {"popular": top_restaurants} if top_restaurants else {},
                "message": f"No preferences found for user {user_id}. Returning popular restaurants."
            }

        preferred_categories = user_prefs["categoryNames"]
        recommendations = {}

        for category in preferred_categories:
            top_restaurants = get_top_restaurants_by_category(category, top_n=3)
            if top_restaurants:
                recommendations[category] = top_restaurants
            else:
                logger.warning(f"No valid restaurants found for category: {category}")

        if not recommendations:
            logger.warning(f"No recommendations generated for user {user_id}. Falling back to popular restaurants.")
            top_restaurants = get_top_restaurants_by_category(top_n=5)
            recommendations["popular"] = top_restaurants

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "message": f"Recommendations generated for user {user_id} based on preferences"
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        return {
            "user_id": user_id,
            "recommendations": {},
            "message": f"Error generating recommendations for user {user_id}: {str(e)}"
        }