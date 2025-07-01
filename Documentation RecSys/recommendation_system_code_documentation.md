# Recommendation System Code Documentation

## Overview

This document provides a detailed, script-by-script explanation of the recommendation system, its architecture, and the flow of data and logic throughout the codebase.

---

## 1. High-Level Architecture

The system is a modular, multi-engine recommendation platform for restaurants and products, using a combination of collaborative filtering, content-based filtering, and fallback strategies. It is orchestrated by a central class and exposed via a Flask API, with Kafka integration for asynchronous processing.

---

## 2. Core Scripts and Their Roles

### A. Orchestration Layer

#### `orchestrator.py` — **RecommendationOrchestrator**
- **Purpose:** Central controller that coordinates all engines to generate recommendations for a user.
- **Logic:**
  1. **Collaborative Recommendation:** Tries to get restaurant recommendations using collaborative filtering.
  2. **Preference Fallback:** If collaborative fails, uses user preferences (from fallback engine).
  3. **Global Popular Fallback:** If no preferences, recommends globally popular restaurants.
  4. **Product Recommendation:** For each recommended restaurant, uses content-based filtering to suggest products.
  5. **MMR Diversification:** Uses Maximal Marginal Relevance (MMR) to diversify restaurant recommendations.
  6. **Feedback Integration:** Adjusts scores based on user likes/dislikes.
  7. **Returns:** A detailed dictionary with recommendations, products, neighbor info, and user preferences.

---

### B. Recommendation Engines

#### `collaborative_engine.py` — **CollaborativeEngine**
- **Purpose:** Implements user-user collaborative filtering for restaurant recommendations.
- **Logic:**
  - Loads user review data and computes user vectors (using BERT).
  - Finds similar users (neighbors) using cosine similarity on preference vectors.
  - Aggregates ratings from neighbors, applies time decay, and recommends restaurants not yet rated by the target user.
  - Can rebuild user profiles and precompute neighbors for efficiency.

#### `content_engine.py` — **ContentEngine**
- **Purpose:** Manages content-based product recommendations.
- **Logic:**
  - Wraps `ContentBasedProductRecommender`.
  - For a given user and restaurant, recommends products based on similarity to previously ordered products, price proximity, product ratings, and sentiment.

#### `content_based_product_recommender.py` — **ContentBasedProductRecommender**
- **Purpose:** Core logic for content-based product recommendations.
- **Logic:**
  - Uses TF-IDF on product descriptions/ingredients.
  - Computes similarity between user's past products and available products.
  - Scores products using a weighted sum of ingredient similarity, price proximity, product rating, and sentiment.
  - Falls back to popular or random products if no history is available.

#### `fallback_engine.py` — **FallbackEngine**
- **Purpose:** Provides fallback recommendations when collaborative/content-based methods fail.
- **Logic:**
  - **Preference Fallback:** Uses user's stated cuisine/category preferences to recommend top restaurants in those categories.
  - **Global Popular:** If no preferences, recommends globally top-rated restaurants.

#### `preference_Recommendation.py`
- **Purpose:** Helper for fallback engine; fetches top restaurants by category or overall, based on user preferences stored in MongoDB.

---

### C. Data and Model Handling

#### `model_handler.py` — **BERTModelHandler**
- **Purpose:** Handles BERT-based vectorization of user reviews for collaborative filtering.

#### `data_loader.py`
- **Purpose:** Utilities for loading MongoDB collections, user data, and resolving user IDs.

#### `mongo_utils.py`
- **Purpose:** Saves generated recommendations to MongoDB for later retrieval.

#### `order_repository.py`, `product_repository.py`, `user_repository.py`
- **Purpose:** Abstraction layers for accessing orders, products, and users from MongoDB.

---

### D. API and Integration

#### `api.py`
- **Purpose:** Flask API exposing endpoints for restaurant and product recommendations.
- **Logic:**
  - `/recommendations/restaurants`: Returns restaurant recommendations for a user.
  - `/recommendations/products`: Returns product recommendations for a user.
  - Stores recommendations in MongoDB and sends events to Kafka.

#### `kafka_producer.py` & `kafka_consumer.py`
- **Purpose:** Integrate with Kafka for asynchronous recommendation requests and logging.

---

### E. Utility and Testing

#### `main_runner.py`
- **Purpose:** Script to run the recommendation pipeline for test users and save results to MongoDB.

#### `test_mmr_vs_no_mmr.py`, `test_weights.py`
- **Purpose:** Scripts to test the effect of MMR and different weighting schemes on recommendations.

#### `precompute_neighbors.py`
- **Purpose:** Precomputes and stores user neighbors for collaborative filtering efficiency.

---

## 3. Recommendation Flow (Step-by-Step)

1. **User Request:** User requests recommendations via API or Kafka.
2. **Orchestrator:** The `RecommendationOrchestrator` receives the request.
3. **Collaborative Filtering:** Tries to recommend restaurants based on similar users' ratings.
4. **Fallbacks:**
   - If collaborative fails, uses user preferences (cuisine/category).
   - If no preferences, recommends globally popular restaurants.
5. **Product Recommendations:** For each recommended restaurant, suggests products using content-based filtering.
6. **MMR Diversification:** Ensures diversity in restaurant recommendations.
7. **Feedback:** Adjusts recommendations based on user likes/dislikes.
8. **Persistence:** Saves recommendations to MongoDB and optionally sends to Kafka.
9. **Response:** Returns recommendations to the user.

---

## 4. Configuration and Deployment

- **`config.yaml`**: Stores model paths, weights, and engine parameters.
- **`docker-compose.yml` & `Dockerfile`**: Define the deployment stack (Flask app, MongoDB, Kafka, Zookeeper).
- **`supervisord.conf`**: Manages process supervision for the app and Kafka consumer.

---

## 5. Summary Table: Script Responsibilities

| Script/File                        | Main Responsibility                                      |
|-------------------------------------|----------------------------------------------------------|
| `orchestrator.py`                   | Central recommendation logic and fallback coordination   |
| `collaborative_engine.py`           | User-user collaborative filtering for restaurants        |
| `content_engine.py`                 | Content-based product recommendation manager             |
| `content_based_product_recommender.py` | Core content-based product recommendation logic      |
| `fallback_engine.py`                | Fallback strategies (preferences, global popular)        |
| `preference_Recommendation.py`      | Fetches top restaurants by user preference               |
| `model_handler.py`                  | BERT-based user vector management                        |
| `data_loader.py`                    | MongoDB and user data utilities                          |
| `mongo_utils.py`                    | Save recommendations to MongoDB                          |
| `order_repository.py`               | Order data access abstraction                            |
| `product_repository.py`             | Product data access abstraction                          |
| `user_repository.py`                | User data access abstraction                             |
| `api.py`                            | Flask API for recommendations                            |
| `kafka_producer.py`/`kafka_consumer.py` | Kafka integration for async processing              |
| `main_runner.py`                    | Batch/test recommendation runner                         |
| `test_mmr_vs_no_mmr.py`/`test_weights.py` | Testing and evaluation scripts                      |
| `precompute_neighbors.py`           | Precomputes user neighbors for collaborative filtering   |

---

## 6. How to Extend or Debug

- **Add new engines:** Implement a new engine class and add it to the orchestrator.
- **Tune weights:** Adjust weights in `config.yaml` for content-based scoring.
- **Debugging:** Use logging (set to DEBUG) to trace recommendation flow and errors.
- **Testing:** Use the provided test scripts to evaluate changes.

---

If you want a more detailed breakdown of any specific script or a diagram of the architecture, let me know! 