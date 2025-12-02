from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path("/data/")

DATA_COLLECTION_ROOT = ASSETS_DIR / "data_collection"

MODEL_PATH = PROJECT_ROOT / "assets" / "best.onnx"
CLASSES_YAML = PROJECT_ROOT / "assets" / "classes.yaml"

DATASET_DIR =  ASSETS_DIR / "duckietown_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

CONF_THRESHOLD = 0.5
STOP_DISTANCE = 0.5
FORWARD_PWM = 0.2

SAVE_EVERY_N_FRAMES = 3
MAX_LOG_IMAGES = 1000
