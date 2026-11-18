"""Project entry point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from insurance_risk_assessment.data import load_data
from insurance_risk_assessment.models import plot_training_history, train_neural_network


def main() -> None:
    parser = argparse.ArgumentParser(description="Insurance risk assessment")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    data = load_data()
    _, history, metrics = train_neural_network(data, epochs=args.epochs)
    plot_training_history(
        history,
        Path(__file__).resolve().parent / "reports" / "training_history.png",
    )
    print(f"R2: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")


if __name__ == "__main__":
    main()
