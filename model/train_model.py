import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickets.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'classifier.pkl')


def train_and_save():
    df = pd.read_csv(DATA_PATH)

    X = df['ticket_text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])

    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Classifier accuracy: {acc:.2f}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {MODEL_PATH}")
    return pipeline


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found, training now...")
        return train_and_save()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    train_and_save()
