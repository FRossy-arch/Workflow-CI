# modelling.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # Load dataset (pastikan file ini ada di folder MLProject)
    df = pd.read_csv("wine-quality-dataset_preprocessing.csv")

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
