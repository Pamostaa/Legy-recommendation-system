import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

try:
    stop_words = set(stopwords.words('english') + stopwords.words('french'))
except:
    stop_words = set(["the", "a", "an", "and", "or", "of", "to", "in", "et", "le", "la", "de", "un", "une"])

class ContentBasedProductRecommender:
    def __init__(self, product_repository, order_repository, weights):
        self.product_repository = product_repository
        self.order_repository = order_repository
        self.weights = weights  # (w1, w2, w3, w4)
        self.df = self._load_products()

    def _load_products(self):
        products = self.product_repository.get_all_products()
        df = pd.DataFrame(products)
        df = df[df["price"].notnull() & df["description"].notnull()]
        df["description"] = df["description"].astype(str)
        df["CleanText"] = df["description"].apply(self._preprocess_text)
        df["PriceValue"] = pd.to_numeric(df["price"], errors='coerce')
        df = df.dropna(subset=["PriceValue"])
        df["Rating"] = 3.5  # default constant if rating is missing

        # Compute sentiment scores
        df["Sentiment"] = df["description"].apply(self._compute_sentiment)

        df = df.reset_index(drop=True)
        return df

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        ps = PorterStemmer()
        tokens = text.split()
        return ' '.join([ps.stem(word) for word in tokens if word not in stop_words])

    def _compute_sentiment(self, text):
        try:
            sentiment_score = vader_analyzer.polarity_scores(text)['compound']
            return sentiment_score
        except Exception as e:
            print(f"[WARNING] Sentiment analysis failed: {e}")
            return 0.0

    def _get_user_product_ids(self, client_id):
        orders = self.order_repository.get_orders_by_client_id(client_id)
        product_ids = []
        for order in orders:
            for item in order.get("items", []):
                pid = item.get("productId")
                if pid:
                    product_ids.append(pid)
        return list(set(product_ids))

    def recommend_for_user(self, client_id: str, restaurant_id: str, top_n=5):
        user_product_ids = self._get_user_product_ids(client_id)
        if not user_product_ids:
            return []

        user_df = self.df[self.df["_id"].astype(str).isin(user_product_ids)]
        if user_df.empty:
            return []

        avg_price = user_df["PriceValue"].mean()

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.df["CleanText"])
        user_vector = vectorizer.transform(user_df["CleanText"])
        mean_user_vector = np.asarray(user_vector.mean(axis=0)).reshape(1, -1)

        self.df["IngredientSimilarity"] = cosine_similarity(mean_user_vector, tfidf_matrix).flatten()
        self.df["PriceProximity"] = 1 - abs(self.df["PriceValue"] - avg_price) / (self.df["PriceValue"].max() - self.df["PriceValue"].min())

        w1, w2, w3, w4 = self.weights
        self.df["FinalScore"] = (
            w1 * self.df["IngredientSimilarity"] +
            w2 * self.df["PriceProximity"] +
            w3 * (self.df["Rating"] / 5.0) +
            w4 * ((self.df["Sentiment"] + 1) / 2)
        )

        filtered_df = self.df[(~self.df["_id"].astype(str).isin(user_product_ids)) & (self.df["restaurant_Id"] == restaurant_id)]
        top_df = filtered_df.sort_values(by="FinalScore", ascending=False).head(top_n)
        return top_df.to_dict(orient="records")
