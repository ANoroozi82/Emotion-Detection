import joblib

model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/model_vectorizer.joblib")

def now_emotion(text):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return prediction

def future_emotion(text):
    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]
    return probabilities
