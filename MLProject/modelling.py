import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    # Load dataset (HARUS ada di folder MLProject)
    df = pd.read_csv("abalone_preprocessing.csv")

    # Feature dan target
    X = df.drop(columns=["Rings"])
    y = df["Rings"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

       
        joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    main()
