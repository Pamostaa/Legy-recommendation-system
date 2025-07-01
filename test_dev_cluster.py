from data_loader import get_mongo_client, get_collections

client = get_mongo_client()
collections = get_collections(client)
user_recs = collections["db"]["user_recommendations"]

# Example recommendation document
doc = {
    "user_id": "test_user_123",
    "type": "restaurant",
    "recommendations": {
        "RecommendedRestaurants": [
            {"_id": "60d5ec49f8d2e4a1b8e7b1a1", "nom": "Test Restaurant", "averageRating": 4.5}
        ]
    },
    "created_at": "2025-07-01T12:00:00Z"
}

user_recs.replace_one({"user_id": "test_user_123", "type": "restaurant"}, doc, upsert=True)
print("Inserted test recommendation for user_id=test_user_123")