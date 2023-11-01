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

    for index, row in df.iterrows():
        # Tokenize the title
        tokens = tokenize_sentences(row['video_name'])
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

    return features_list, labels_list


# sample_text = "Bilan du premier tour des législatives : la déroute des candidats portés par la complosphère"
# tokenized_sample = tokenize_sentences(sample_text)
# print(tokenized_sample)
# Diagnostic to find the mismatch between number of words and number of labels
# mismatched_rows_updated = []
# for index, row in df.iterrows():
#     num_words = len(tokenize_sentences(row['video_name']))
#     num_labels = len(eval(row['is_name']))
#     if num_words != num_labels:
#         mismatched_rows_updated.append((index, row['video_name'], row['is_name'], num_words, num_labels))
#         # mismatched_rows_updated.append(index)
# # Display the mismatched rows
# print(mismatched_rows_updated)
# print(len(mismatched_rows_updated))

def make_features(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
        X = df["video_name"].apply(preprocess_text_P1)

    elif task == "is_name":
        # Drop bugged row due of ":"
        indices_to_drop = [75, 95, 108, 159, 179, 182, 231, 360, 377, 392, 404, 410, 417, 483, 507, 541, 693, 742, 763, 829, 843, 844, 877, 881, 992]
        df = df.drop(indices_to_drop)

        X, y = preprocess_text_P2_matching_labels(df)

    elif task == "find_comic_name":
        y = df["comic_name"]

    else:
        raise ValueError("Unknown task")

    return X, y
