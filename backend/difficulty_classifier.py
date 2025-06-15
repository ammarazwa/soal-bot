import joblib
import os

_model = None

def load_model():
    global _model
    if _model is None:
        model_path = os.path.join("model", "difficulty_model.pkl")
        _model = joblib.load(model_path)

def predict_difficulty(sentence):
    load_model()
    return _model.predict([sentence])[0]
