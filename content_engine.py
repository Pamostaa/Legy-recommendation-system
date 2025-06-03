from content_based_product_recommender import ContentBasedProductRecommender

class ContentEngine:
    def __init__(self, product_repository, order_repository, weights=(0.3, 0.3, 0.2, 0.2)):
        self.product_repository = product_repository
        self.order_repository = order_repository
        self.cbf = ContentBasedProductRecommender(product_repository, order_repository, weights)
