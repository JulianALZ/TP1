import pandas as pd


def make_dataset(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"No file found at path: {filename}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data in file: {filename}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file: {filename}")
        return None


def explore_data(data):
    print("First few rows of the data:")
    print(data.head())

    print("\nDescriptive statistics of the data:")
    print(data.describe(include='all'))
