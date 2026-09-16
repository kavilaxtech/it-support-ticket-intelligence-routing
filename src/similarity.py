import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TicketSimilarityEngine:

    def __init__(self, ticket_texts):

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.ticket_matrix = self.vectorizer.fit_transform(
            ticket_texts
        )

    def find_similar(self, query, top_k=5):

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarity_scores = cosine_similarity(
            query_vector,
            self.ticket_matrix
        ).flatten()

        indices = np.argsort(
            similarity_scores
        )[::-1][:top_k]

        scores = similarity_scores[indices]

        return indices, scores