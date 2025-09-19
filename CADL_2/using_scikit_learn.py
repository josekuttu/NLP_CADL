import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -------------------------
# 1. Load Dataset
# -------------------------
# Replace 'your_dataset.csv' with your file name
df = pd.read_csv("dataset/tweets.csv")

# Focus on the text column 'content'
texts = df['content'].astype(str)

# -------------------------
# 2. Bag-of-Words Representation
# -------------------------
bow_vectorizer = CountVectorizer(max_features=10)  # limit to 10 features for display
bow_matrix = bow_vectorizer.fit_transform(texts)

print("\n=== Bag-of-Words ===")
print("Feature Names:", bow_vectorizer.get_feature_names_out())
print("BoW Matrix (first 5 rows):\n", bow_matrix.toarray()[:5])

# -------------------------
# 3. TF-IDF Representation
# -------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print("\n=== TF-IDF ===")
print("Feature Names:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix (first 5 rows):\n", tfidf_matrix.toarray()[:5])
