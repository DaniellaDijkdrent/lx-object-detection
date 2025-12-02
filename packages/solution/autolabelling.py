import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image

from config import DATASET_DIR, TRAIN_DIR, VAL_DIR, CLASSES_YAML

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_classes():
    with open(CLASSES_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    items = sorted(cfg["classes"].items(), key=lambda kv: int(kv[0]))
    return [name for _, name in items]


def xyxy_to_yolo_line(bbox, w, h, class_id):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def init_sam3(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes()
    bpe_path = Path("/code/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    model.to(device)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.3, device=device)
    return processor, classes


def detect_sam3(img_path, state):
    processor, classes = state
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    with torch.no_grad():
        st = processor.set_image(image)
        detections = []
        for class_id, prompt in enumerate(classes):
            out = processor.set_text_prompt(state=st, prompt=prompt)
            boxes = out.get("boxes")
            scores = out.get("scores")
            if boxes is None or scores is None:
                continue
            for b, s in zip(boxes, scores):
                s = float(s)
                if s < processor.confidence_threshold:
                    continue
                detections.append(
                    {
                        "bbox": b.tolist(),
                        "score": s,
                        "class_id": int(class_id),
                    }
                )
    return detections, (w, h)


def auto_label_folder(data_dir: Path, split: str = "train", backend: str = "sam3"):
    if split == "train":
        split_dir = TRAIN_DIR
    elif split == "val":
        split_dir = VAL_DIR
    else:
        raise ValueError("split must be 'train' or 'val'")

    images_out = split_dir / "images"
    labels_out = split_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    csv_path = DATASET_DIR / f"autolabel_{backend}_{split}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["backend", "image_path", "class_id", "score", "x1", "y1", "x2", "y2"])

    if backend != "sam3":
        raise ValueError(f"unsupported backend: {backend}")

    state = init_sam3()

    image_paths = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    for img_path in image_paths:
        dets, (w, h) = detect_sam3(img_path, state)
        out_img = images_out / img_path.name
        if out_img != img_path:
            out_img.write_bytes(img_path.read_bytes())

        yolo_lines = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            score = d["score"]
            class_id = d["class_id"]
            writer.writerow(
                ["sam3", str(out_img), class_id, score, x1, y1, x2, y2]
            )
            yolo_lines.append(
                xyxy_to_yolo_line(d["bbox"], w, h, class_id)
            )

        label_path = labels_out / (img_path.stem + ".txt")
        if yolo_lines:
            label_path.write_text("\n".join(yolo_lines))
        else:
            if label_path.exists():
                label_path.unlink()

    csv_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--backend", type=str, default="sam3", choices=["sam3"]) #TODO: add support for "owlv2"
    args = parser.parse_args()
    auto_label_folder(args.data_dir, args.split, args.backend)


if __name__ == "__main__":
    main()
