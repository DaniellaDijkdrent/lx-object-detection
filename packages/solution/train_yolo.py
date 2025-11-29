import argparse
from pathlib import Path
from ultralytics import YOLO

from config import DATASET_DIR, MODEL_PATH, CLASSES_YAML


def train_yolo(epochs: int, imgsz: int, batch: int, data_yaml: Path):
    model = YOLO("yolo11n.yaml") # training from scratch

    runs_root = DATASET_DIR / "runs"
    run_name = "duckietown_detection"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        project=str(runs_root),
        name=run_name,
    )

    best = runs_root / run_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)

    model = YOLO(str(best))
    onnx_tmp = DATASET_DIR / "detector.onnx" #may need to separate the export from train
    # my assumption about run_name_dir doesn't hold - unless I force ultralytics to overwrite old runs

    model.export(
        format="onnx",
        opset=18,
        imgsz=(480, 640), #fixed for the momen
        simplify=True,
        dynamic=False,
        nms=True,         
        half=True,        
        batch=1,
        optimize=False,
        export_path=str(onnx_tmp),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(onnx_tmp.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--data_yaml", type=Path, required=True)
    args = parser.parse_args()

    train_yolo(args.epochs, args.imgsz, args.batch, args.data_yaml)


if __name__ == "__main__":
    main()
