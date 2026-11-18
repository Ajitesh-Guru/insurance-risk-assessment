import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insurance_risk_assessment.data import load_data


if __name__ == "__main__":
    data = load_data()
    X = data.drop(columns=["charges"])
    y = data["charges"]
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["sex", "smoker", "region"],
            ),
            ("numerical", StandardScaler(), ["age", "bmi", "children"]),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"R2: {r2_score(y_test, predictions):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.2f}")
