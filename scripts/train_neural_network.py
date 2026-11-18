from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurance_risk_assessment.data import load_data
from insurance_risk_assessment.models import plot_training_history, train_neural_network


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data = load_data()
    _, history, metrics = train_neural_network(data)
    plot_training_history(history, project_root / "reports" / "training_history.png")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
