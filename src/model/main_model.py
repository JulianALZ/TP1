from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


class DumpableModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        dump(self.model, filename_output)

    def load(self, filename_input):
        self.model = load(filename_input)


def make_model(task, dumpable=True):
    if task == "is_comic_video":
        model = Pipeline([
            ("count_vectorizer", CountVectorizer()),
            ("classifier", RandomForestClassifier()),

            # Random Forest
            # ("classifier", RandomForestClassifier()),    # 89-91 %

            # Logistic Regression
            # ("classifier", LogisticRegression()),   # ~ 91%

            # Support Vector Machine (SVM)
            # ("classifier", SVC()),  # ~92%

            # Multinomial Naive Bayes
            # ("classifier", MultinomialNB()),
        ])

    elif task == "is_name":
        model = Pipeline([
            ("dict_vectorizer", DictVectorizer(sparse=True)),
            # ("classifier", MultinomialNB()),  # 95.20305499785123%
            # ("classifier", LogisticRegression(max_iter=1000)),  #  96.13070494874881%
            # ("classifier", RandomForestClassifier()),  # 95.34805079281293%
            ("classifier", SVC())  # 96.14742036872373%
        ])
    else:
        raise ValueError("Unknown task")

    if dumpable:
        return DumpableModel(model)
    else:
        return model
