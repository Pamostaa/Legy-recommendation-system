class OrderRepository:
    def __init__(self, orders_col):
        self.orders_col = orders_col

    def get_orders_by_client_id(self, client_id):
        return list(self.orders_col.find({"client.clientId": client_id}))

    def user_has_history_with_restaurant(self, client_id, restaurant_id):
        return self.orders_col.count_documents({
            "client.clientId": client_id,
            "items.restaurantId": restaurant_id
        }) > 0

    def has_any_orders(self, client_id):
        return self.orders_col.count_documents({"clientId": client_id}) > 0

    def get_reviews_by_user(self, user_id):
        return list(self.orders_col.find({"User": user_id}))