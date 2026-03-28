import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickets.csv')

_vectorizer = None
_tfidf_matrix = None
_df = None


def _load():
    global _vectorizer, _tfidf_matrix, _df
    if _vectorizer is None:
        _df = pd.read_csv(DATA_PATH)

        # Initialize TF-IDF
        _vectorizer = TfidfVectorizer()
        _tfidf_matrix = _vectorizer.fit_transform(_df['ticket_text'])


def get_similar_tickets(query: str, top_k: int = 3):
    """Return top_k most similar tickets and the best similarity score."""
    _load()

    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _tfidf_matrix)[0]

    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'ticket_text': _df.iloc[idx]['ticket_text'],
            'category': _df.iloc[idx]['category'],
            'resolution': _df.iloc[idx]['resolution'],
            'similarity': float(sims[idx])
        })

    best_score = float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0

    return results, best_score