import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load Training Dataset

df = pd.read_csv('dataset/Train.csv')

# Show first rows

print(df.head())

# First column = text
# Second column = sentiment

X_text = df.iloc[:, 0]

y = df.iloc[:, 1]

# Convert text into numbers

vectorizer = CountVectorizer(stop_words='english')

X = vectorizer.fit_transform(X_text)

# Train Model

model = MultinomialNB()

model.fit(X, y)

# Save trained model

joblib.dump(model, 'sentiment_model.pkl')

joblib.dump(vectorizer, 'vectorizer.pkl')

print('Professional NLP Model Trained Successfully 🚀')