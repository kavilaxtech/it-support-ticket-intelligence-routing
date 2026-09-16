from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_text_classifier():
    """
    Create an NLP classification pipeline.

    TF-IDF converts text into numerical features.
    Logistic Regression performs classification.
    """

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=8000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return pipeline


def train_models(X_train, y_category, y_issue, y_priority):

    category_model = build_text_classifier()
    issue_model = build_text_classifier()
    priority_model = build_text_classifier()

    category_model.fit(X_train, y_category)

    issue_model.fit(X_train, y_issue)

    priority_model.fit(X_train, y_priority)

    return (
        category_model,
        issue_model,
        priority_model
    )