from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INSTANCE_FOLDER = PROJECT_ROOT / "instances"
DATA_FOLDER = PROJECT_ROOT / "data"
DATASETS_FOLDER = PROJECT_ROOT / "datasets"
MODELS_FOLDER = PROJECT_ROOT / "models"
HYPERPARAMETERS_FOLDER = MODELS_FOLDER / "hyperparameters"

FRG_PATH = PROJECT_ROOT / "src" / "frg"