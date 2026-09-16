from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_text_classifier():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=8000)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

def train_models(X, y_category, y_issue, y_priority):
    models = [build_text_classifier(), build_text_classifier(), build_text_classifier()]
    models[0].fit(X, y_category)
    models[1].fit(X, y_issue)
    models[2].fit(X, y_priority)
    return tuple(models)
