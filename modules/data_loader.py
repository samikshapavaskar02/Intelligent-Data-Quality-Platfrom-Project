import pandas as pd

def load_dataset(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    elif file.name.endswith(".json"):
        return pd.read_json(file)

    else:
        return None