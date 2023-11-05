import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.preprocessing import FunctionTransformer

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


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# Transformateur personnalisé pour effectuer la prédiction avec le modèle de vidéo comique
# class ComicVideoFilter(BaseEstimator, TransformerMixin):
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         # La prédiction renvoie 1 pour les vidéos comiques et 0 sinon
#         return self.model.predict(X)
#
#
# # Transformateur personnalisé pour effectuer le prétraitement P2 et la prédiction de noms
# class NameFeatureExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X, y=None):
#         # Aucun ajustement requis ici, donc on retourne simplement self
#         return self
#
#     def transform(self, X):
#         # Appliquer le prétraitement P2 seulement pour les entrées comiques
#         X_comic = [x for x, is_comic in zip(X, self.comic_filter_) if is_comic]
#         if not X_comic:  # S'il n'y a pas de vidéos comiques, retourner une liste vide
#             return []
#
#         features_list, _ = preprocess_text_P2_matching_labels(X_comic)
#         predictions = self.model.predict(features_list)
#         reshaped_predictions = reshape_predictions_using_position(predictions, features_list)
#         return reshaped_predictions
#
#     def set_params(self, **kwargs):
#         super().set_params(**kwargs)
#         self.comic_filter_ = kwargs.get('comic_filter', None)
#         return self




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
            # ("classifier", MultinomialNB()),  # Accuracy: 95.20% / Precision: 93.16% / Recall: 95.20% / F1 Score: 94.08%
            # ("classifier", LogisticRegression(max_iter=1000)),  #  Accuracy: 96.13% / Precision: 92.44% / Recall: 96.13% / F1 Score: 94.25%
            # ("classifier", RandomForestClassifier()),  # Accuracy: 95.51% / Precision: 93.77% / Recall: 95.51% / F1 # Score: 94.46%
            ("classifier", SVC())  # Accuracy: 96.15% / Precision: 92.44% / Recall: 96.15% / F1 Score: 94.26%

        ])
    elif task == "find_comic_name":
        # comic_video_filter = ComicVideoFilter(model=comic_video_model)
        # name_feature_extractor = NameFeatureExtractor(model=is_name_model)
        # comic_video_filter = P3(model=comic_video_model)
        # name_feature_extractor = P3(model=is_name_model)

        # La pipeline exécutera d'abord la prédiction de vidéo comique, puis le prétraitement P2 et la prédiction de noms
        # model = Pipeline([
        #     ('comic_video_filter', comic_video_filter),
        #     ('name_feature_extractor', name_feature_extractor)
        # ])
        model = Pipeline([

        ])

    else:
        raise ValueError("Unknown task")

    if dumpable:
        return DumpableModel(model)
    else:
        return model

#
# # COPY OF SMAE FUNCTION OF OTHER FILE DUE OF BUG
# def preprocess_text_P2_matching_labels(df):
#     """
#     Preprocess the text for the "is_name" task by extracting relevant features.
#     Tokenization is adjusted to match the labels' tokenization.
#     """
#     features_list = []
#     labels_list = []
#
#     for index, row in df.iterrows():
#         # Tokenize the title
#         tokens = tokenize_sentences(row['video_name'])
#         individual_labels = ast.literal_eval(row['is_name'])
#
#         for i, token in enumerate(tokens):
#             previous_word = tokens[i - 1] if i > 0 else None
#             next_word = tokens[i + 1] if i < len(tokens) - 1 else None
#
#             # Extract features for the token
#             features = {
#                 "is_capitalized": token[0].isupper() if token else False,
#                 "length": len(token),
#                 "position": i,
#                 "is_previous_capitalized": previous_word[
#                     0].isupper() if previous_word and previous_word != "" else False,
#                 "is_next_capitalized": next_word[0].isupper() if next_word and next_word != "" else False,
#             }
#
#             # Append the features and the corresponding label to the lists
#             features_list.append(features)
#             labels_list.append(individual_labels[i])
#
#     return features_list, labels_list
#
#
# def reshape_predictions_using_position(predictions, X_test):
#     reshaped_predictions = []
#     sentence_predictions = []
#
#     for i, feature_dict in enumerate(X_test):
#         sentence_predictions.append(predictions[i])
#         if i < len(X_test) - 1 and X_test[i + 1]['position'] == 0:
#             reshaped_predictions.append(sentence_predictions)
#             sentence_predictions = []
#     # Ajouter les prédictions de la dernière phrase
#     if sentence_predictions:
#         reshaped_predictions.append(sentence_predictions)
#
#     return reshaped_predictions
#
#
# def tokenize_sentences(text):
#     """Tokenize the text based on the rules provided by the professor."""
#
#     # Split words based on spaces
#     words = text.split()
#     final_words = []
#     for word in words:
#         if word == "-":
#             final_words.append(word)
#         elif "-" in word and not word.startswith("-") and not word.endswith("-"):
#             final_words.append(word)
#         elif word.endswith(":"):
#             final_words.append(word)
#         elif "'" in word:
#             parts = word.split("'")
#             final_words.append(parts[0] + "'")
#             final_words.extend(parts[1:])
#         else:
#             final_words.append(word)
#     return final_words
