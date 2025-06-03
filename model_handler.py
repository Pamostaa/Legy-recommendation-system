# model_handler.py
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)


class BERTModelHandler:
    def __init__(self, model_path, vectors_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        with open(vectors_path, "rb") as f:
            self.id_to_vector = pickle.load(f)
        self.vectors_path = vectors_path

    def compute_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = softmax(logits, dim=1).squeeze().numpy()
        return probs

    def save_vectors(self):
        with open(self.vectors_path, "wb") as f:
            pickle.dump(self.id_to_vector, f)
