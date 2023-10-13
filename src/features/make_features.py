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
