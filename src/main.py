import click
import numpy as np
from sklearn.model_selection import cross_val_score
from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model, DumpableModel
from sklearn.model_selection import train_test_split


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--test_size", default=0.2, type=float, help="...")#ajout
def train(task, input_filename, model_dump_filename, test_size):
    df = make_dataset(input_filename)
    # explore_data(df)

    # mismatched_rows = []
    #
    # for index, row in df.iterrows():
    #     num_words = len(row['video_name'].split())
    #     num_labels = len(eval(row['is_name']))
    #     if num_words != num_labels:
    #         mismatched_rows.append((index, row['video_name'], row['is_name'], num_words, num_labels))
    #
    # # Display the mismatched rows
    # print(len(mismatched_rows))


    X, y = make_features(df, task)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)#ajout

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
    X_test, _ = make_features(test_data, task)  # Using the provided task
    
    # 3. Predict using the model
    predictions = model.predict(X_test)

    # 4. Save predictions to a file
    with open(output_filename, "w") as f:
        for video_name,is_name,is_comic,comic_name, prediction in zip(test_data["video_name"],test_data["is_name"],test_data["is_comic"],test_data["comic_name"], predictions):
            f.write(f"{video_name},{is_name},{is_comic},{comic_name}, {prediction}\n")

    print(f"Predictions with video names saved to {output_filename}")

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model(task=task, dumpable=False)
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")
    print(f"Got accuracy {100 * np.mean(scores)}%")
    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
