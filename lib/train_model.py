import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_and_save_model(file_path, model_path):
    data_set = pd.read_csv(file_path)

    data_set['Clean_Text'] = data_set['Clean_Text'].fillna("")

    vectorizer = TfidfVectorizer()
    X_features = vectorizer.fit_transform(data_set['Clean_Text'])
    emotion = data_set['Emotion']

    X_train, X_test, y_train, y_test = train_test_split(X_features, emotion, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, model_path.replace(".joblib", "_vectorizer.joblib"))


train_and_save_model("data/emotion_dataset.csv", "models/model.joblib")
