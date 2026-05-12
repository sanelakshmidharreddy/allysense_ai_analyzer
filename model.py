import joblib

# Load saved model

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def predict_sentiment(text):

    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)[0]

    probability = model.predict_proba(transformed_text)

    confidence = round(max(probability[0]) * 100, 2)

    return prediction, confidence