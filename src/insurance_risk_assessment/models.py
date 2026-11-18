from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data import PROJECT_ROOT
from .preprocessing import build_preprocessor


MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "neural_network.h5"


def train_neural_network(
    data: pd.DataFrame,
    epochs: int = 500,
    batch_size: int = 64,
    random_state: int = 42,
) -> tuple[tf.keras.Model, object, dict]:
    X = data.drop(columns=["charges"])
    y = data["charges"]
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="sigmoid"),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
    )

    y_pred = model.predict(X_test, verbose=0)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    return model, history, metrics


def plot_training_history(history: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history.history["mae"], label="Training MAE")
    axes[1].plot(history.history["val_mae"], label="Validation MAE")
    axes[1].set(title="Mean absolute error", xlabel="Epoch", ylabel="MAE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
