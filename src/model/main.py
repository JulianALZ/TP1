# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# from joblib import dump, load
#
#
# def make_model(dumpable=True):
#     model = Pipeline([
#         ("count_vectorizer", CountVectorizer()),
#         ("random_forest", RandomForestClassifier()),
#     ])
#     if dumpable:
#         return DumpableModel(model)
#     else:
#         return model
#
#
# class DumpableModel:
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X, y=None):
#         return self.model.fit(X, y)
#
#     def predict(self, X):
#         return self.model.predict(X)
#
#     def dump(self, filename_output):
#         # Save the model parameters and anything else necessary for predictions
#         dump(self.model, filename_output)
#
#     def load(self, filename_input):
#         # Load the model parameters
#         self.model = load(filename_input)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from joblib import dump, load


def make_model(task, dumpable=True):
    if task == "is_comic_video":
        model = Pipeline([
            ("count_vectorizer", CountVectorizer()),
            ("classifier", RandomForestClassifier()),
            
        # Random Forest
        # ("random_forest", RandomForestClassifier()),    # 89-91 %

        # Logistic Regression
        # ("classifier", LogisticRegression()),   # ~ 91%

        # Support Vector Machine (SVM)
        # ("classifier", SVC()),  # ~92%

        # Multinomial Naive Bayes
        #("classifier", MultinomialNB()),
        ])
    elif task == "is_name":
        model = Pipeline([
            ("count_vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB())
        ])
    else:
        raise ValueError("Unknown task")
    
    if dumpable:
        return DumpableModel(model)
    else:
        return model


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
