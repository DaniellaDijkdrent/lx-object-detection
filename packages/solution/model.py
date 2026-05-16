#!/usr/bin/env python3

import numpy as np
import cv2
import onnxruntime as ort

from dt_computer_vision.camera.types import Pixel
from duckietown_messages.actuators.differential_pwm import DifferentialPWM

from solution.config import (
    MODEL_PATH,
    CONF_THRESHOLD,
    STOP_DISTANCE,
    FORWARD_PWM,
)


class MLModel:
    def __init__(self):
        print("Initializing MLModel")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found at: {MODEL_PATH}")

        print("AVAILABLE PROVIDERS:", ort.get_available_providers())

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name

        self.net_h = int(inp.shape[2])
        self.net_w = int(inp.shape[3])

        print(f"Model loaded: {self.net_w}x{self.net_h}")

        self.ground_projector = None

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    def _preprocess(self, img_bgr):

        # resize
        img = cv2.resize(img_bgr, (self.net_w, self.net_h))

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # IMPORTANT:
        # model expects FLOAT16
        img = img.astype(np.float16) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    # -----------------------------
    # INFERENCE
    # -----------------------------
    def _run_detector(self, img_bgr):

        x = self._preprocess(img_bgr)

        print("INPUT SHAPE:", x.shape)
        print("INPUT DTYPE:", x.dtype)
        print("INPUT RANGE:", x.min(), x.max())

        out = self.session.run(None, {self.input_name: x})

        print("NUM OUTPUTS:", len(out))

        for i, o in enumerate(out):
            print(
                f"OUTPUT {i}: shape={o.shape}, "
                f"min={o.min():.4f}, max={o.max():.4f}"
            )

        return out

    # -----------------------------
    # YOLOv8 PARSER
    # -----------------------------
    def _parse_detections(self, outputs):

        preds = outputs[0]

        if preds is None or len(preds) == 0:
            return []

        # remove batch dimension
        preds = preds[0]

        # transpose if needed
        # (84,8400) -> (8400,84)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.transpose(1, 0)

        detections = []

        for p in preds:

            if len(p) < 6:
                continue

            # YOLO format:
            # x,y,w,h,class_scores...
            x, y, w, h = p[:4]

            class_scores = p[4:]

            cls = np.argmax(class_scores)
            conf = class_scores[cls]

            if conf < CONF_THRESHOLD:
                continue

            # xywh -> xyxy
            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            detections.append((
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(conf),
                int(cls),
            ))

        print(f"Parsed {len(detections)} detections")

        return detections

    # -----------------------------
    # STOP LOGIC
    # -----------------------------
    def _should_stop(self, detections):

        if len(detections) == 0:
            return False

        stop = False

        for x1, y1, x2, y2, score, cls in detections:

            print(f"Detection CONF={score:.2f}")

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            pix = Pixel(x=cx, y=cy)

            try:
                vec = self.ground_projector.camera.pixel2vector(pix)
                ground_point = self.ground_projector.vector2ground(vec)

                dist = np.linalg.norm(ground_point)

                print(f"Distance: {dist:.3f}m")

                if dist < STOP_DISTANCE:
                    stop = True

            except Exception as e:
                print("Projection failed:", e)
                stop = True

        return stop

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def set_ground_projector(self, gp):
        self.ground_projector = gp

    def get_wheel_velocities_from_image(self, img):

        try:
            outputs = self._run_detector(img)
            detections = self._parse_detections(outputs)

        except Exception as e:
            print("Inference error:", e)

            return [
                DifferentialPWM(left=0.0, right=0.0),
                None,
            ]

        print("DETECTIONS:", detections)

        stop = self._should_stop(detections)

        if stop:
            return [
                DifferentialPWM(left=0.0, right=0.0),
                detections,
            ]

        return [
            DifferentialPWM(
                left=FORWARD_PWM,
                right=FORWARD_PWM,
            ),
            detections,
        ]