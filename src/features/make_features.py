import re
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


# def preprocess_text_P2(text):
#     """
#     Preprocess the text for the "is_name" task by extracting relevant features.
#     """
#     # Tokenize the text
#     words = text.split()
#
#     # Extract features for each word
#     features_list = []
#     for i, word in enumerate(words):
#         previous_word = words[i - 1] if i > 0 else None
#         next_word = words[i + 1] if i < len(words) - 1 else None
#         features = extract_word_features_P2(word, i, previous_word, next_word)
#         features_list.append(features)
#
#     return features_list


def tokenize_matching_labels(text):
    """Tokenize the text to match the label's tokenization."""
    # Split based on spaces first
    words = text.split()
    final_words = []
    for word in words:
        # If an apostrophe is found in the word, split it into two
        if "'" in word:
            parts = word.split("'")
            final_words.append(parts[0] + "'")
            final_words.extend(parts[1:])
        else:
            final_words.append(word)
    return final_words


def preprocess_text_P2_matching_labels(text):
    """
    Preprocess the text for the "is_name" task by extracting relevant features.
    Tokenization is adjusted to match the labels' tokenization.
    """
    # Tokenize the text to match labels
    words = tokenize_matching_labels(text)

    # Extract features for each word
    features_list = []
    for i, word in enumerate(words):
        previous_word = words[i - 1] if i > 0 else None
        next_word = words[i + 1] if i < len(words) - 1 else None
        features = extract_word_features_P2(word, i, previous_word, next_word)
        features_list.append(features)

    return features_list


def extract_word_features_P2(text, position, previous_word, next_word):
    """
    Extract features for a given word with safety checks.
    """
    return {
        "is_capitalized": text[0].isupper() if text else False,
        "length": len(text),
        "position": position,
        "is_previous_capitalized": previous_word[0].isupper() if previous_word else False,
        "is_next_capitalized": next_word[0].isupper() if next_word else False,
    }

def make_features(df, task):
    # Diagnostic to find the mismatch between number of words and number of labels
    mismatched_rows_updated = []

    for index, row in df.iterrows():
        num_words = len(tokenize_matching_labels(row['video_name']))
        num_labels = len(eval(row['is_name']))
        if num_words != num_labels:
            mismatched_rows_updated.append((index, row['video_name'], row['is_name'], num_words, num_labels))

    # Display the mismatched rows
    print(mismatched_rows_updated)
    if task == "is_comic_video":
        y = df["is_comic"]
        X = df["video_name"].apply(preprocess_text_P1)
    elif task == "is_name":
        y = df["is_name"]
        X = df["video_name"].apply(preprocess_text_P2_matching_labels)
    elif task == "find_comic_name":
        y = df["comic_name"]

    else:
        raise ValueError("Unknown task")

    return X, y






# EXO2

# def preprocess_text_for_features(text):
#     try:
#         # Check if the input is string type
#         if isinstance(text, str):
#             # Convert to lowercase but keep original for some feature extraction
#             original_text = text[:]
#             text = text.lower()
#
#             # Only remove unwanted characters, but keep punctuations for feature extraction
#             text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
#
#             return original_text, text
#         else:
#             print("Input is not a string")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#
#
# def extract_aggregated_features(text):
#     # Preprocess the text
#     original_sentence, sentence = preprocess_text_for_features(text)
#     sentence_words = sentence.split()
#     original_sentence_words = original_sentence.split()
#
#     # Calculate the features
#     total_words = len(sentence_words)
#     capitalized_words = sum(1 for word in original_sentence_words if word[0].isupper())
#     has_exclamation = 1 if "!" in original_sentence else 0
#     has_question = 1 if "?" in original_sentence else 0
#
#     # Define the features dictionary
#     features = {
#         "total_words": total_words,
#         "capitalized_words": capitalized_words,
#         "has_exclamation": has_exclamation,
#         "has_question": has_question
#     }
#
#     return features
#
#
# # Test the function with a sample text
# # extract_aggregated_features("L'humeur de Marina Rollman!")
#
#
# def make_aggregated_features_and_labels(df, task):
#     X = []
#     y = []
#
#     for index, row in df.iterrows():
#         features = extract_aggregated_features(row['video_name'])
#         # Convert the label string representation to a list and aggregate
#         label_list = eval(row[task])
#         label = 1 if any(label_list) else 0
#
#         X.append(features)
#         y.append(label)
#
#     return X, y
