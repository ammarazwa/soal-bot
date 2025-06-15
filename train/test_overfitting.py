import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model
model = joblib.load(os.path.join("..","model", "difficulty_model.pkl"))

# Load data
df = pd.read_excel(os.path.join("..","model", "kalimat_difficulty_100.xlsx"))
X = df["kalimat"]
y = df["label"]

# Split ulang (agar bisa bandingkan langsung)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Prediksi dan evaluasi test set
y_pred = model.predict(X_test)

print("\n🎯 Prediksi vs Label Asli:")
for kalimat, pred, true in zip(X_test, y_pred, y_test):
    print(f"- Kalimat: {kalimat}\n  → Prediksi: {pred}, Label Asli: {true}\n")

print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation (5-fold)
print("\n🔁 Cross Validation (cv=5):")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

cv_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])
cv_scores = cross_val_score(cv_model, X, y, cv=5)
print("Fold scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# Uji prediksi pada kalimat baru
print("\n🧪 Prediksi pada Kalimat Baru:")
new_sentences = [
    "Jelaskan apa yang dimaksud dengan sistem komputer.",
    "Langkah-langkah membuat chatbot dengan machine learning cukup kompleks.",
    "Apa manfaat dari algoritma sorting dalam struktur data?"
]
for s in new_sentences:
    print(f"- {s} → Prediksi: {model.predict([s])[0]}")
