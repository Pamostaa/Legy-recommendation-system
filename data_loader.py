import pandas as pd
from pymongo import MongoClient
import os
import logging
import yaml
from bson import ObjectId

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_mongo_client(uri=None):
    if uri is None:
        uri = os.getenv("MONGO_URI")
        logging.debug(f"MONGO_URI from environment: {uri}")
        if not uri:
            # Try to load from config.yaml
            try:
                with open("config.yaml") as f:
                    config = yaml.safe_load(f)
                    uri = config.get("mongo_uri")
                    logging.debug(f"MONGO_URI from config.yaml: {uri}")
            except Exception as e:
                logging.error(f"Could not load mongo_uri from config.yaml: {str(e)}")
        if not uri:
            # Fallback to default cluster URI
            uri = "mongodb://legy:legy123@mongo:27017/leggydb?authSource=admin"
            logging.debug(f"Using fallback MONGO_URI: {uri}")
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        logging.error(f"Invalid URI scheme: URI must begin with 'mongodb://' or 'mongodb+srv://', got {uri}")
        raise ValueError(f"Invalid URI scheme: {uri}")
    try:
        client = MongoClient(uri)
        client.admin.command("ping")
        logging.info("MongoDB client initialized successfully")
        logging.debug(f"Available databases: {client.list_database_names()}")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize MongoDB client: {str(e)}", exc_info=True)
        raise

def get_collections(client):
    try:
        db = client["leggydb"]
        logging.debug(f"Using database: {db.name}")
        collections = db.list_collection_names()
        logging.debug(f"Available collections in leggydb: {collections}")
        expected = ["avis-restaurant", "users", "restaurants", "products", "orders", "Category", "restaurant_reactions", "user_neighbors"]
        missing = [col for col in expected if col not in collections]
        if missing:
            logging.warning(f"Missing collections: {missing}")

        return {
            "db": db,
            "avis-restaurant": db["avis-restaurant"],
            "users": db["users"],
            "restaurants": db["restaurants"],
            "products": db["products"],
            "orders": db["orders"],
            "categories": db["Category"],
            "restaurant_reactions": db["restaurant_reactions"],
            "user_neighbors": db["user_neighbors"]
        }
    except Exception as e:
        logging.error(f"Error getting collections: {str(e)}", exc_info=True)
        raise

def reload_users(users_col):
    try:
        users_data = list(users_col.find())
        if not users_data:
            logging.warning("No users found in users collection")
            return pd.DataFrame(), {}, {}
        
        users = pd.DataFrame(users_data)
        if "_id" in users.columns:
            users["_id"] = users["_id"].apply(str)
        if "user_id" not in users.columns:
            users["user_id"] = users["_id"]
            logging.debug("Added user_id column to users DataFrame")
        name_to_id = {row["username"]: row.get("user_id", str(row["_id"])) for _, row in users.iterrows()}
        id_to_name = {row.get("user_id", str(row["_id"])): row["username"] for _, row in users.iterrows()}
        logging.debug(f"Loaded {len(users)} users: {users[['_id', 'user_id', 'username']].to_dict('records')}")
        return users, name_to_id, id_to_name
    except Exception as e:
        logging.error(f"Error reloading users: {str(e)}", exc_info=True)
        return pd.DataFrame(), {}, {}

def resolve_user(users_df, name_to_id, id_to_name, user_input):
    logging.debug(f"Resolving user input: '{user_input}'")
    try:
        if user_input in id_to_name:
            logging.debug(f"Found user by ID: {user_input} -> {id_to_name[user_input]}")
            return user_input, id_to_name[user_input]
        elif user_input in name_to_id:
            logging.debug(f"Found user by username: {user_input} -> {name_to_id[user_input]}")
            return name_to_id[user_input], user_input
        match = users_df[users_df["username"].str.lower() == str(user_input).lower()]
        if not match.empty:
            user_id = str(match.iloc[0]["user_id"])
            username = match.iloc[0]["username"]
            logging.debug(f"Found user by username (case-insensitive): {user_input} -> {user_id}, {username}")
            return user_id, username
        match = users_df[users_df["user_id"] == str(user_input)]
        if not match.empty:
            user_id = str(match.iloc[0]["user_id"])
            username = match.iloc[0]["username"]
            logging.debug(f"Found user by user_id in DataFrame: {user_input} -> {user_id}, {username}")
            return user_id, username
        match = users_df[users_df["_id"] == str(user_input)]
        if not match.empty:
            user_id = str(match.iloc[0]["user_id"])
            username = match.iloc[0]["username"]
            logging.debug(f"Found user by _id in DataFrame: {user_input} -> {user_id}, {username}")
            return user_id, username
        logging.warning(f"User not found for input: '{user_input}'. Available IDs: {list(id_to_name.keys())[:5]}...")
        return None, None
    except Exception as e:
        logging.error(f"Error resolving user input '{user_input}': {str(e)}", exc_info=True)
        return None, None