import re
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ast

# Uncomment this line when running locally
nltk.download('stopwords')
nltk.download('wordnet')  # Download WordNet data

stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()


def preprocess_text_P1(text):
    try:
        # Check if the input is string type
        if isinstance(text, str):
            # Convert text to lowercase
            text = text.lower()

            # Remove accented characters
            text = unidecode.unidecode(text)

            # Keep only alphabets and spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Remove stopwords
            text = ' '.join(word for word in text.split() if word not in stop_words)

            # Lemmatization
            text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

            return text
        else:
            print("Input is not a string")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def tokenize_sentences(text):
    """Tokenize the text based on the rules provided by the professor."""

    # Split words based on spaces
    words = text.split()
    final_words = []
    for word in words:
        if word == "-":
            final_words.append(word)
        elif "-" in word and not word.startswith("-") and not word.endswith("-"):
            final_words.append(word)
        elif word.endswith(":"):
            final_words.append(word)
        elif "'" in word:
            parts = word.split("'")
            final_words.append(parts[0] + "'")
            final_words.extend(parts[1:])
        else:
            final_words.append(word)
    return final_words


def preprocess_text_P2_matching_labels(df):
    """
    Preprocess the text for the "is_name" task by extracting relevant features.
    Tokenization is adjusted to match the labels' tokenization.
    """
    features_list = []
    labels_list = []
    tokens_list = []

    for index, row in df.iterrows():
        # Tokenize the title
        tokens = tokenize_sentences(row['video_name'])
        tokens_list.append(tokens)
        individual_labels = ast.literal_eval(row['is_name'])

        for i, token in enumerate(tokens):
            previous_word = tokens[i - 1] if i > 0 else None
            next_word = tokens[i + 1] if i < len(tokens) - 1 else None

            # Extract features for the token
            features = {
                "is_capitalized": token[0].isupper() if token else False,
                "length": len(token),
                "position": i,
                "is_previous_capitalized": previous_word[
                    0].isupper() if previous_word and previous_word != "" else False,
                "is_next_capitalized": next_word[0].isupper() if next_word and next_word != "" else False,
            }

            # Append the features and the corresponding label to the lists
            features_list.append(features)
            labels_list.append(individual_labels[i])

    return features_list, labels_list, tokens_list


def make_features(df, task, comic_video_model=None, is_name_model=None):
    tokens_list = None
    if task == "is_comic_video":
        y = df["is_comic"]
        X = df["video_name"].apply(preprocess_text_P1)

    elif task == "is_name":
        X, y, tokens_list = preprocess_text_P2_matching_labels(df)

    elif task == "find_comic_name":
        y = df["comic_name"]


    else:
        raise ValueError("Unknown task")

    return X, y, tokens_list


