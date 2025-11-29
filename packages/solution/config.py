from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path("/data/duckiebot_images") #PROJECT_ROOT / "assets" -- ideally should be assets but perhaps not in project_root

DATA_COLLECTION_ROOT = ASSETS_DIR / "data_collection_logs" #TODO: Ask

MODEL_PATH = PROJECT_ROOT / "assets" / "detector.onnx"
CLASSES_YAML = ASSETS_DIR / "classes.yaml"

DATASET_DIR =  ASSETS_DIR / "duckietown_object_detection_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

CONF_THRESHOLD = 0.5
STOP_DISTANCE_M = 0.5
FORWARD_PWM = 0.1

SAVE_EVERY_N_FRAMES = 3
MAX_LOG_IMAGES = 1000
