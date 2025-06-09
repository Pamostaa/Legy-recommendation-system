# 🍽️ Food Delivery Recommendation System

This project implements a modular, scalable recommendation system for a food delivery mobile app, combining collaborative filtering, content-based filtering, sentiment analysis, and diversification (MMR).

## 🚀 Features

- ✅ Collaborative filtering with user vectors
- ✅ Content-based filtering (ingredient + price matching)
- ✅ Sentiment analysis using VADER
- ✅ Maximal Marginal Relevance (MMR) for diversified recommendations
- ✅ Time decay weighting for recent interactions
- ✅ Dockerized for easy deployment (<2GB image)

## 🏗 Technologies

- Python (Flask, Pandas, Scikit-learn, Transformers, Torch)
- MongoDB (user, restaurant, review storage)
- Redis (optional, for caching)
- Docker (production-ready)

## 🔧 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/recommendation-system.git
   cd recommendation-system
