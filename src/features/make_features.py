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


def preprocess_text(text):
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


def make_features(df, task):
    y = get_output(df, task)

    X = df["video_name"].apply(preprocess_text)

    print(X[4])  # print specific data

    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y

def preprocess_text_for_features(text):
    try:
        # Check if the input is string type
        if isinstance(text, str):
            # Convert to lowercase but keep original for some feature extraction
            original_text = text[:]
            text = text.lower()

            # Only remove unwanted characters, but keep punctuations for feature extraction
            text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)

            return original_text, text
        else:
            print("Input is not a string")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def extract_aggregated_features(text):
    # Preprocess the text
    original_sentence, sentence = preprocess_text_for_features(text)
    sentence_words = sentence.split()
    original_sentence_words = original_sentence.split()
    
    # Calculate the features
    total_words = len(sentence_words)
    capitalized_words = sum(1 for word in original_sentence_words if word[0].isupper())
    has_exclamation = 1 if "!" in original_sentence else 0
    has_question = 1 if "?" in original_sentence else 0

    # Define the features dictionary
    features = {
        "total_words": total_words,
        "capitalized_words": capitalized_words,
        "has_exclamation": has_exclamation,
        "has_question": has_question
    }
    
    return features

# Test the function with a sample text
extract_aggregated_features("L'humeur de Marina Rollman!")

def make_aggregated_features_and_labels(df, task):
    X = []
    y = []

    for index, row in df.iterrows():
        features = extract_aggregated_features(row['video_name'])
        # Convert the label string representation to a list and aggregate
        label_list = eval(row[task])
        label = 1 if any(label_list) else 0
        
        X.append(features)
        y.append(label)

    return X, y

