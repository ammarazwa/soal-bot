import joblib
import os

_model = None
_vectorizer = None

def load_model():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        model_path = os.path.join("model", "kmeans_topic.pkl")
        _model, _vectorizer = joblib.load(model_path)

def predict_cluster(sentence):
    load_model()
    X = _vectorizer.transform([sentence])
    return _model.predict(X)[0]
