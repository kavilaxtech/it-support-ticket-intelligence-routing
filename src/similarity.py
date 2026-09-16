import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TicketSimilarityEngine:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def find_similar(self, query, top_k=5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores[idx]
