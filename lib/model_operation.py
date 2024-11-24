import joblib as jl

model = jl.load(open("./models/emotion_classifier_pipe_lr.pkl","rb"))

def nowEmotion(docx):
    results = model.predict(docx)
    return results[0]

def futureEmotion(docx):
    results = model.predict_proba(docx)
    return results