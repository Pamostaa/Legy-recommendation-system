import pandas as pd

def save_recommendations_to_mongo(db, user_id, result):
    if "Recommendations" not in result or not result["Recommendations"]:
        return

    db["user_recommendations"].replace_one(
        {"user_id": user_id},
        {
            "user_id": user_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "recommendations": result["Recommendations"],
            "products": result["Products"],
            "neighbor_ids": result.get("Neighboring Users (IDs)", []),
            "neighbor_names": result.get("Neighboring Users (Names)", []),
            "neighbor_preferences": result.get("Neighbor Preferences", {}),
            "restaurants_rated_by_target_user": result.get("Restaurants Rated by Target User", []),
            "restaurants_rated_by_neighbors": result.get("Restaurants Rated by Neighbors", {}),
            "target_user_preference": result.get("Target User Preference", {}),
            "message": result.get("Message", "")
        },
        upsert=True
    )
    print(f"✅ Recommendations saved for user {user_id}")
