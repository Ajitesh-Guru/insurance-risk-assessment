from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurance_risk_assessment.analysis import generate_eda_report
from insurance_risk_assessment.data import load_data


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    generate_eda_report(load_data(), project_root / "reports")
