
# 📊 Smart Grocery Recommendation System – Hybrid Model

This project implements a **hybrid machine learning-based recommender system** for a grocery store dataset. It leverages both **collaborative filtering (KNN-based)** and **content-based filtering (TF-IDF)** to generate accurate and personalized product suggestions.

## ✅ Objectives

- To explore customer purchase behavior and transaction trends.
- To build a recommendation engine that combines collaborative and content-based techniques.
- To evaluate model performance using industry-standard metrics.
- To support personalized product recommendations for both existing and new users.

## 🧾 Dataset Description

The dataset includes grocery transactions in the following format:

- `User_id`: Unique ID of each customer.
- `Date`: Date of transaction.
- `itemDescription`: Name of the item purchased.
- `year`, `month`, `day`, `day_of_week`: Extracted date parts.

## 🛠️ Features & Techniques

- **Exploratory Data Analysis (EDA):** Transaction frequency, top products, re-purchase patterns, time-based trends.
- **Preprocessing:** Time conversion, user-item matrix creation, text normalization.
- **Content-Based Filtering:**
  - TF-IDF vectorization of user purchase history.
  - Cosine similarity to recommend similar products not yet purchased.
- **Collaborative Filtering:**
  - KNN using user-item matrix and cosine similarity.
  - Nearest neighbor purchases used for personalized suggestions.
- **Hybrid Approach:**
  - Weighted combination of both techniques.
  - Allows for fine-tuning between content vs. behavior-driven results.

## 🧪 Model Evaluation

The recommendation system is evaluated using:

- **Precision@K** – Relevance of top-K predictions.
- **Recall@K** – Coverage of relevant items.
- **F1-Score@K** – Harmonic mean of precision and recall.
- **MRR (Mean Reciprocal Rank)** – Ranking of first relevant item.

## 📌 How to Run

1. Load the dataset:
   ```python
   df = pd.read_csv("Groceries data train.csv")
   ```

2. Preprocess & split data into training and testing.

3. Run:
   ```python
   evaluate_hybrid_model(...)
   ```

4. Get hybrid recommendations for any user:
   ```python
   hybrid_recommendation(...)
   ```

## 📈 Example Output

```
📊 Hybrid Recommender Evaluation (alpha=0.5, top-5):
✅ Precision@5: 0.4020
✅ Recall@5:    0.3112
✅ F1-Score@5:  0.3512
✅ MRR:         0.4937
```

## 📚 Requirements

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📌 Future Enhancements

- Incorporate product metadata (e.g., categories, prices).
- Handle cold-start users via rule-based or popularity models.
- Deploy as API with Flask or Streamlit.
