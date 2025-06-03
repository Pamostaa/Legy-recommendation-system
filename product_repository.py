class ProductRepository:
    def __init__(self, products_col):
        self.products_col = products_col

    def get_all_products(self):
        return list(self.products_col.find({}))

    def get_products_by_restaurant(self, restaurant_id, limit=5):
        return list(self.products_col.find({"restaurant_Id": str(restaurant_id)}).limit(limit))