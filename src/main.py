import click
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main_model import make_model, DumpableModel
from sklearn.model_selection import train_test_split


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--test_size", default=0.2, type=float, help="...")  # ajout
def train(task, input_filename, model_dump_filename, test_size):
    df = make_dataset(input_filename)
    # explore_data(df)

    X, y = make_features(df, task)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) #ajout

    model = make_model(task=task, dumpable=True)

    model.fit(X, y)

    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    # 1. Load the trained model
    model = DumpableModel(None)  # Initialize with no model
    model.load(model_dump_filename)

    # 2. Load and preprocess the test data
    test_data = make_dataset(input_filename)
    X_test, y = make_features(test_data, task)  # Using the provided task

    print("X ///////////", X_test[0:10])
    print("y ///////////", y[0:10])
    print("LEN X =", len(X_test))
    print("LEN y =", len(y))

    # 3. Predict using the model
    predictions = model.predict(X_test)
    print("predictions = ", predictions)

    reshaped_pred = reshape_predictions_using_position(predictions, X_test)

    # 4. Save predictions to a file
    with open(output_filename, "w", encoding='utf-8') as f:
        for video_name, is_name, is_comic, comic_name, tokens, prediction in zip(test_data["video_name"],
                                                                                 test_data["is_name"],
                                                                                 test_data["is_comic"],
                                                                                 test_data["comic_name"],
                                                                                 test_data["tokens"], reshaped_pred):
            f.write(f"{video_name},{is_name},{is_comic},{comic_name}, {tokens}, {prediction}\n")

    print(f"Predictions with video names saved to {output_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model(task=task, dumpable=False)
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Effectuer une validation croisée pour obtenir des prédictions
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Calculer l'accuracy, la précision, le rappel et le F1-score
    accuracy = accuracy_score(y, y_pred)  # Pourcentage de prédictions correctes
    precision = precision_score(y, y_pred, average='weighted')  # Pourcentage de prédictions positives correctes
    recall = recall_score(y, y_pred, average='weighted')  # Pourcentage de vraies positives par rapport à tous les vrais échantillons
    f1 = f1_score(y, y_pred, average='weighted')  # Moyenne harmonique de précision et rappel

    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1 Score: {100 * f1:.2f}%")

    return accuracy, precision, recall, f1


# def evaluate_model(model, X, y):
#     # Scikit learn has function for cross validation
#     scores = cross_val_score(model, X, y, scoring="accuracy")
#     print(f"Got accuracy {100 * np.mean(scores)}%")
#     return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


def reshape_predictions_using_position(predictions, X_test):
    reshaped_predictions = []
    sentence_predictions = []

    for i, feature_dict in enumerate(X_test):
        sentence_predictions.append(predictions[i])
        if i < len(X_test) - 1 and X_test[i + 1]['position'] == 0:
            reshaped_predictions.append(sentence_predictions)
            sentence_predictions = []
    # Ajouter les prédictions de la dernière phrase
    if sentence_predictions:
        reshaped_predictions.append(sentence_predictions)

    return reshaped_predictions


if __name__ == "__main__":
    cli()
