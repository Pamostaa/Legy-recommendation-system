import pandas as pd

class UserRepository:
    def __init__(self, users_col):
        self.users_col = users_col

    def get_user_by_id(self, user_id):
        return self.users_col.find_one({"_id": user_id})

    def get_all_users(self):
        return list(self.users_col.find({}))

    def reload_users(self):
        users_data = list(self.users_col.find())
        if not users_data:
            print("[WARNING] No users found in Users collection")
            return [], {}, {}

        users_df = pd.DataFrame(users_data)
        users_df["_id"] = users_df["_id"].astype(str)
        name_to_id = dict(zip(users_df["User"], users_df["_id"]))
        id_to_name = dict(zip(users_df["_id"], users_df["User"]))
        return users_df, name_to_id, id_to_name
