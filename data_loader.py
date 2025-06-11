# data_loader.py
import pandas as pd
from pymongo import MongoClient

def get_mongo_client():
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    return MongoClient(config['mongo_uri'])


def get_collections(client):
    db = client["recommender_db"]
    return {
        "db": db,
        "reviews": db["Reviews"],
        "users": db["Users"],
        "restaurents": db["Restaurents"],
        "products": db["Products"],
        "orders": db["Orders"],
        "categories": pd.DataFrame(list(db["restaurant_categories"].find())),
        "RestaurentReaction" : db["RestaurentReaction"]
    }

def reload_users(users_col):
    users_data = list(users_col.find())
    if not users_data:
        print("[WARNING] No users found in Users collection")
        return pd.DataFrame(), {}, {}

    users = pd.DataFrame(users_data)
    users["_id"] = users["_id"].astype(str)
    name_to_id = dict(zip(users["User"], users["_id"]))
    id_to_name = dict(zip(users["_id"], users["User"]))
    return users, name_to_id, id_to_name
