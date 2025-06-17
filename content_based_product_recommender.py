import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

vader_analyzer = SentimentIntensityAnalyzer()
try:
    stop_words = set(stopwords.words('english') + stopwords.words('french'))
except:
    stop_words = set(["the", "a", "an", "and", "or", "of", "to", "in", "et", "le", "la", "de", "un", "une"])

class ContentBasedProductRecommender:
    def __init__(self, product_repository, order_repository, weights):
        self.product_repository = product_repository
        self.order_repository = order_repository
        self.weights = weights
        self.df = self._load_products()

    def _load_products(self):
        products = self.product_repository.get_all_products()
        df = pd.DataFrame(products)
        if df.empty:
            logging.error("No products loaded from repository")
            return pd.DataFrame()
        df = df[df["price"].notnull() & df["description"].notnull() & df["restaurant_Id"].notnull()]
        df["description"] = df["description"].astype(str)
        df["_id"] = df["_id"].astype(str)
        df["restaurant_Id"] = df["restaurant_Id"].astype(str)
        df["CleanText"] = df["description"].apply(self._preprocess_text)
        df["PriceValue"] = pd.to_numeric(df["price"], errors='coerce')
        df = df.dropna(subset=["PriceValue"])
        df["Rating"] = 3.5  # Default rating
        df["Sentiment"] = df["description"].apply(self._compute_sentiment)
        df = df.reset_index(drop=True)
        logging.debug(f"Loaded {len(df)} products")
        return df

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        ps = PorterStemmer()
        tokens = text.split()
        return ' '.join([ps.stem(word) for word in tokens if word not in stop_words])

    def _compute_sentiment(self, text):
        try:
            return vader_analyzer.polarity_scores(text)['compound']
        except Exception as e:
            logging.warning(f"Sentiment analysis failed: {e}")
            return 0.0

    def _get_user_product_ids(self, client_id: str) -> List[str]:
        orders = self.order_repository.get_orders_by_client_id(client_id)
        product_ids = []
        for order in orders:
            for item in order.get("items", []):
                pid = item.get("productId")
                if pid:
                    product_ids.append(str(pid))
        product_ids = list(set(product_ids))
        logging.debug(f"User {client_id} has {len(product_ids)} product IDs: {product_ids}")
        return product_ids

    def recommend_for_user(self, client_id: str, restaurant_id: Optional[str], top_n: int = 5) -> List[dict]:
        if self.df.empty:
            logging.warning("No products available in database")
            return []

        user_product_ids = self._get_user_product_ids(client_id)
        logging.debug(f"Processing recommendations for user {client_id}, restaurant {restaurant_id}")

        # Case 1: Specific restaurant
        if restaurant_id:
            rest_df = self.df[self.df["restaurant_Id"] == str(restaurant_id)]
            if rest_df.empty:
                logging.debug(f"No products for restaurant {restaurant_id}")
                return []
            if not user_product_ids:
                logging.debug(f"No order history for {client_id}, using random fallback for restaurant {restaurant_id}")
                return self._random_fallback(rest_df, top_n)
            return self._recommend_similar_products(rest_df, user_product_ids, top_n)

        # Case 2: No specific restaurant (fallback)
        if not user_product_ids:
            logging.debug(f"No order history for {client_id}, using random fallback")
            return self._random_fallback(self.df, top_n)

        # Case 2a: Recommend similar products from all restaurants
        logging.debug(f"Recommending similar products for {client_id} from all restaurants")
        return self._recommend_similar_products(self.df, user_product_ids, top_n)

    def _recommend_similar_products(self, target_df: pd.DataFrame, user_product_ids: List[str], top_n: int) -> List[dict]:
        user_df = self.df[self.df["_id"].isin(user_product_ids)]
        if user_df.empty:
            logging.debug(f"No matching products for user history, using random fallback")
            return self._random_fallback(target_df, top_n)

        avg_price = user_df["PriceValue"].mean()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.df["CleanText"])
        user_vector = vectorizer.transform(user_df["CleanText"])
        mean_user_vector = np.asarray(user_vector.mean(axis=0)).reshape(1, -1)

        target_df["IngredientSimilarity"] = cosine_similarity(mean_user_vector, tfidf_matrix[target_df.index]).flatten()
        price_range = target_df["PriceValue"].max() - target_df["PriceValue"].min()
        target_df["PriceProximity"] = 1 - abs(target_df["PriceValue"] - avg_price) / (price_range if price_range > 0 else 1)
        target_df["PriceProximity"] = target_df["PriceProximity"].fillna(0.5)

        w1, w2, w3, w4 = self.weights
        target_df["FinalScore"] = (
            w1 * target_df["IngredientSimilarity"] +
            w2 * target_df["PriceProximity"] +
            w3 * (target_df["Rating"] / 5.0) +
            w4 * ((target_df["Sentiment"] + 1) / 2) +
            np.random.uniform(0, 0.01, len(target_df))  # Break ties
        )

        filtered_df = target_df[~target_df["_id"].isin(user_product_ids)]
        logging.debug(f"Filtered {len(filtered_df)} products for client")

        if filtered_df.empty:
            logging.debug(f"No filtered products, using random fallback")
            return self._random_fallback(target_df, top_n)

        logging.debug(f"Scores: {filtered_df[['_id', 'name', 'IngredientSimilarity', 'PriceProximity', 'FinalScore']].to_dict('records')}")
        top_df = filtered_df.sort_values(by="FinalScore", ascending=False).head(top_n)
        return top_df.to_dict(orient="records")

    def _random_fallback(self, df: pd.DataFrame, top_n: int) -> List[dict]:
        if df.empty:
            return []
        if "category" in df.columns:
            top_df = df.groupby("category").apply(
                lambda x: x.sample(n=min(1, len(x)), random_state=None)
            ).reset_index(drop=True)
            if len(top_df) >= top_n:
                return top_df.head(top_n).to_dict(orient="records")
        top_df = df.sample(n=min(top_n, len(df)), random_state=None)
        logging.debug(f"Random fallback selected {len(top_df)} products")
        return top_df.to_dict(orient="records")