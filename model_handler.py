# model_handler.py
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import yaml
import os

with open('config.yaml') as f:
    config = yaml.safe_load(f)

class BERTModelHandler:
    def __init__(self, model_path, vectors_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.vectors_path = vectors_path
        self.id_to_vector = self.load_vectors()

    def compute_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = softmax(logits, dim=1).squeeze().numpy()
        return probs

    def save_vectors(self):
        try:
            with open(self.vectors_path, "wb") as f:
                pickle.dump(self.id_to_vector, f)
            print(f"[INFO] Saved vectors to {self.vectors_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save vectors: {e}")

    def load_vectors(self):
        if not os.path.exists(self.vectors_path):
            print(f"[WARNING] Vectors file not found at {self.vectors_path}. Starting with empty vectors.")
            return {}

        if os.path.getsize(self.vectors_path) == 0:
            print(f"[WARNING] Vectors file is empty at {self.vectors_path}. Starting with empty vectors.")
            return {}

        try:
            with open(self.vectors_path, "rb") as f:
                vectors = pickle.load(f)
                if isinstance(vectors, dict):
                    print(f"[INFO] Loaded {len(vectors)} vectors from {self.vectors_path}")
                    return vectors
                else:
                    print(f"[WARNING] Unexpected vector format in {self.vectors_path}. Starting empty.")
                    return {}
        except Exception as e:
            print(f"[ERROR] Could not load vectors: {e}. Starting with empty.")
            return {}
