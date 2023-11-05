import ast

import click
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main_model import make_model, DumpableModel
from sklearn.model_selection import train_test_split

from NER_model import NER_Model


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
    # Drop bugged row due of ":"
    indices_to_drop = [75, 95, 108, 159, 179, 182, 231, 360, 377, 392, 404, 410, 417, 483, 507, 541, 693, 742, 763, 829, 843, 844, 877, 881, 992]
    df = df.drop(indices_to_drop)
    # explore_data(df)

    if task == "NER_MODEL":
        # Instancier le processeur de texte
        ner_model = NER_Model()

        # Traiter le dataframe pour extraire les entités et ajouter une nouvelle colonne avec les résultats NER
        df_processed = ner_model.process_dataframe(df)
        print(df_processed)

        return

    else :
        X, y, tokens_list = make_features(df, task)

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
    # Load and preprocess the test data
    test_data = make_dataset(input_filename)

    # Drop bugged row due of ":"
    indices_to_drop = [75, 95, 108, 159, 179, 182, 231, 360, 377, 392, 404, 410, 417, 483, 507, 541, 693, 742, 763, 829, 843, 844, 877, 881, 992]
    test_data = test_data.drop(indices_to_drop)

    if task == "NER_MODEL":
        # Instancier le processeur de texte
        ner_model = NER_Model()

        # Traiter le dataframe pour extraire les entités et ajouter une nouvelle colonne avec les résultats NER
        df_processed = ner_model.process_dataframe(test_data)

        # Filtre les vidéos qui contiennent le nom spécifié
        name_to_search = "Frédéric Fromet"  # Remplacez par le nom que vous recherchez
        df_videos_with_name = ner_model.find_videos_with_name(df_processed, name_to_search)

        print(df_videos_with_name)
        df_videos_with_name.to_csv(output_filename, index=False)
        print(f"Predictions with NER NAMES saved to {output_filename}")

        return

    else :
        model = DumpableModel(None)  # Initialize with no model
        model.load(model_dump_filename)

        X_test, y, tokens_list = make_features(test_data, task)  # Using the provided task

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
                                                                                     test_data["tokens"],
                                                                                     reshaped_pred):
                f.write(f"{video_name},{is_name},{is_comic},{comic_name}, {tokens}, {prediction}\n")

        print(f"Predictions with video names saved to {output_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    df = make_dataset(input_filename)

    # Drop bugged row due of ":"
    indices_to_drop = [75, 95, 108, 159, 179, 182, 231, 360, 377, 392, 404, 410, 417, 483, 507, 541, 693, 742, 763, 829, 843, 844, 877, 881, 992]
    df = df.drop(indices_to_drop)

    if task == "find_comic_name":
        X, y = P3(df)
        print(X)
        return calculate_accuracy(X, y)
    else:
        X, y, tokens_list = make_features(df, task)

        model = make_model(task=task, dumpable=False)

        return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Effectuer une validation croisée pour obtenir des prédictions
    y_pred = cross_val_predict(model, X, y, cv=5)

    accuracy = accuracy_score(y, y_pred)  # Pourcentage de prédictions correctes
    precision = precision_score(y, y_pred, average='weighted')  # Pourcentage de prédictions positives correctes
    recall = recall_score(y, y_pred,
                          average='weighted')  # Pourcentage de vraies positives par rapport à tous les vrais échantillons
    f1 = f1_score(y, y_pred, average='weighted')  # Moyenne harmonique de précision et rappel

    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1 Score: {100 * f1:.2f}%")

    return accuracy, precision, recall, f1


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


def P3(df):
    model = DumpableModel(None)  # Initialize with no model
    model.load('models/model_SVM.json')
    X_model_is_comic, y, tokens_list = make_features(df, task="is_comic_video")
    predictions = model.predict(X_model_is_comic)
    df["result_model_is_comic"] = predictions
    df2 = df[df['result_model_is_comic'] == 1]
    # print(df2)

    model = DumpableModel(None)  # Initialize with no model
    model.load('models_p2/model_RDF.json')
    X_model_is_name, y, tokens_list = make_features(df2, task="is_name")
    predictions = model.predict(X_model_is_name)
    reshaped_pred = reshape_predictions_using_position(predictions, X_model_is_name)
    extracted_tokens = extract_names(tokens_list, reshaped_pred)
    list_values = df2["comic_name"].tolist()
    transformed_list = []
    for item in list_values:
        # Évaluer la chaîne en tant que liste Python
        evaluated_item = ast.literal_eval(item)

        # Ajouter l'élément à la liste transformée si non vide, sinon ajouter None
        transformed_list.append(evaluated_item[0] if evaluated_item else None)

    return extracted_tokens,  transformed_list


def extract_names(tokens_list, mask_list):
    # Cette liste contiendra tous les tokens concaténés où le masque correspondant est un '1',
    # ou "None" si aucun '1' n'est trouvé dans la liste de masques.
    concatenated_tokens = []

    # Parcourir les listes de tokens et de masques.
    for tokens, mask in zip(tokens_list, mask_list):
        # Utiliser list comprehension pour extraire les tokens où le masque est '1'.
        tokens_with_ones = [token for token, flag in zip(tokens, mask) if flag == 1]
        # Concaténer les tokens avec un espace s'il y en a plus d'un dans la même liste.
        # Si aucun token avec un '1' n'est trouvé, ajouter "None" à la liste.
        concatenated_token = ' '.join(tokens_with_ones) if tokens_with_ones else None
        concatenated_tokens.append(concatenated_token)

    return concatenated_tokens


def calculate_accuracy(predicted, actual):
    # Vérifier que les deux listes ont la même longueur
    if len(predicted) != len(actual):
        raise ValueError("Les listes doivent avoir la même longueur.")

    # Calculer le nombre de prédictions correctes
    correct_predictions = sum(p == a for p, a in zip(predicted, actual))

    # Calculer l'accuracy
    accuracy = correct_predictions / len(actual)
    print(accuracy)

cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
