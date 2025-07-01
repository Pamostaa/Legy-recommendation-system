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
import os
from typing import List, Optional

NLTK_DATA = os.getenv("NLTK_DATA", "/app/nltk_data")
nltk.data.path.append(NLTK_DATA)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logging.info("Downloading NLTK stopwords")
    nltk.download('stopwords', download_dir=NLTK_DATA, quiet=True)

vader_analyzer = SentimentIntensityAnalyzer()
try:
    stop_words = set(stopwords.words('english') + stopwords.words('french'))
except Exception as e:
    logging.warning(f"Failed to load stopwords: {e}")
    stop_words = set(["the", "a", "an", "and", "or", "of", "to", "in", "et", "le", "la", "de", "un", "une"])

class ContentBasedProductRecommender:
    def __init__(self, product_repository, order_repository, weights, Category):
        self.product_repository = product_repository
        self.order_repository = order_repository
        self.weights = weights
        # Load category mapping
        categories = list(Category.find())
        self.category_id_to_name = {str(cat['_id']): cat['name'] for cat in categories}
        self.df = self._load_products()

    def _load_products(self):
        products = self.product_repository.get_all_products()
        df = pd.DataFrame(products)
        if df.empty:
            logging.error("No products loaded from repository")
            return pd.DataFrame()
        if "pricePostCom" not in df.columns:
            logging.error("No 'pricePostCom' field found in products collection. Please check your data.")
            return pd.DataFrame()
        df = df[df["pricePostCom"].notnull() & df["description"].notnull() & df["restaurantId"].notnull()]
        df["description"] = df["description"].astype(str)
        df["_id"] = df["_id"].astype(str)
        df["restaurantId"] = df["restaurantId"].astype(str)
        # Add category_name column
        if 'category' in df.columns:
            df['category_name'] = df['category'].apply(lambda cid: self.category_id_to_name.get(str(cid), 'Unknown'))
        df["CleanText"] = df["description"].apply(self._preprocess_text)
        df["PriceValue"] = pd.to_numeric(df["pricePostCom"], errors='coerce')
        df = df.dropna(subset=["PriceValue"])
        df["Rating"] = 3.5
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

        if restaurant_id:
            rest_df = self.df[self.df["restaurantId"] == str(restaurant_id)]
            if rest_df.empty:
                logging.debug(f"No products for restaurant {restaurant_id}")
                return []
            if not user_product_ids:
                logging.debug(f"No order history for {client_id}, using random fallback for restaurant {restaurant_id}")
                return self._random_fallback(rest_df, top_n, restaurant_id)
            return self._recommend_similar_products(rest_df, user_product_ids, top_n)

        if not user_product_ids:
            logging.debug(f"No order history for {client_id}, using random fallback")
            return self._random_fallback(self.df, top_n, None)

        logging.debug(f"Recommending similar products for {client_id} from all restaurants")
        return self._recommend_similar_products(self.df, user_product_ids, top_n)

    def _recommend_similar_products(self, target_df: pd.DataFrame, user_product_ids: List[str], top_n: int) -> List[dict]:
        user_df = self.df[self.df["_id"].isin(user_product_ids)]
        if user_df.empty:
            logging.debug(f"No matching products for user history, using random fallback")
            return self._random_fallback(target_df, top_n, target_df["restaurantId"].iloc[0] if not target_df.empty else None)

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
            np.random.uniform(0, 0.01, len(target_df))
        )

        filtered_df = target_df[~target_df["_id"].isin(user_product_ids)]
        logging.debug(f"Filtered {len(filtered_df)} products for client")

        if filtered_df.empty:
            logging.debug(f"No filtered products, using random fallback")
            return self._random_fallback(target_df, top_n, target_df["restaurantId"].iloc[0] if not target_df.empty else None)

        logging.debug(f"Scores: {filtered_df[['_id', 'name', 'IngredientSimilarity', 'PriceProximity', 'FinalScore']].to_dict('records')}")
        top_df = filtered_df.sort_values(by="FinalScore", ascending=False).head(top_n)
        return top_df.to_dict(orient="records")

    def _random_fallback(self, df: pd.DataFrame, top_n: int, restaurant_id: Optional[str]) -> List[dict]:
        if df.empty:
            return []
        # Filter by restaurant_id if provided
        if restaurant_id:
            df = df[df["restaurantId"] == str(restaurant_id)]
            if df.empty:
                logging.debug(f"No products available for restaurant {restaurant_id} in fallback")
                return []
        # Prioritize popular products by Rating or order frequency
        if "Rating" in df.columns:
            top_df = df.sort_values(by="Rating", ascending=False).head(top_n * 2)
        else:
            top_df = df
        if "category_name" in top_df.columns:
            top_df = top_df.groupby("category_name").apply(
                lambda x: x.sample(n=min(1, len(x)), random_state=None)
            ).reset_index(drop=True)
            if len(top_df) >= top_n:
                return top_df.head(top_n).to_dict(orient="records")
        top_df = top_df.sample(n=min(top_n, len(top_df)), random_state=None)
        logging.debug(f"Random fallback selected {len(top_df)} products")
        return top_df.to_dict(orient="records")