from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_path: Path = Path("/workspaces/nrg_interview_prep/data")
    height_data_file: Path = data_path / "height_data.csv"
    survival_data_file: Path = data_path / "survival_data.csv"

    trained_height_model_path = data_path / "height_model.pkl"
    trained_survival_model_path = data_path / "survival_model.pkl"
    survival_model_stats_path = data_path / "survival_model_stats.json"
