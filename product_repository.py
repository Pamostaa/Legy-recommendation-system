import pandas as pd
from typing import List
import logging

class ProductRepository:
    def __init__(self, products_col):
        self.products_col = products_col

    def get_all_products(self):
        return list(self.products_col.find({}))

    def get_products_by_restaurant(self, restaurant_id, limit=5):
        return list(self.products_col.find({"restaurantId": str(restaurant_id)}).limit(limit))

    def _load_products(self):
        products = self.get_all_products()
        df = pd.DataFrame(products)
        if 'category' in df.columns:
            df['category_name'] = df['category'].apply(lambda cid: self.category_id_to_name.get(str(cid), 'Unknown'))
        return df

    def _random_fallback(self, df: pd.DataFrame, top_n: int) -> List[dict]:
        if df.empty:
            return []
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