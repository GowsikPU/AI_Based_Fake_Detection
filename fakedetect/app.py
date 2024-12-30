from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# Prepare the model
def train_model():
    # Load datasets
    real_news_path = r'D:\fake news 2\True.csv'
    fake_news_path = r'D:\fake news 2\Fake.csv'

    # Load and label datasets
    real_news = pd.read_csv(real_news_path)
    fake_news = pd.read_csv(fake_news_path)

    if 'text' not in real_news.columns or 'text' not in fake_news.columns:
        raise ValueError("Datasets must have a 'text' column containing the news content.")

    real_news['label'] = 1  # Real news
    fake_news['label'] = 0  # Fake news

    # Combine datasets
    news_data = pd.concat([real_news, fake_news], ignore_index=True)

    # Handle missing values
    news_data.dropna(subset=['text'], inplace=True)

    # Prepare data
    X = news_data['text']
    y = news_data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Define a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)),
        ('clf', LogisticRegression(solver='liblinear', random_state=7))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Test the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

    return pipeline

# Train the model on app startup
model = train_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text")
    if not news_text:
        return jsonify({"error": "No input text provided."})

    prediction = model.predict([news_text])[0]
    result = "Real" if prediction == 1 else "Fake"
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default port is 8000
    app.run(host="0.0.0.0", port=port, debug=True)
