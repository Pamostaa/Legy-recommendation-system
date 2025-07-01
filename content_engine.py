from content_based_product_recommender import ContentBasedProductRecommender
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

class ContentEngine:
    def __init__(self, product_repository, order_repository, Category, weights=tuple(config["weights"])):
        self.product_repository = product_repository
        self.order_repository = order_repository
        self.Category = Category
        self.cbf = ContentBasedProductRecommender(product_repository, order_repository, weights, Category)

    def recommend_for_user(self, client_id: str, restaurant_id: str, top_n=5):
        return self.cbf.recommend_for_user(client_id, restaurant_id, top_n)