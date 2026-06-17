#!/usr/bin/env python3

import numpy as np
import cv2
from pathlib import Path
import onnxruntime as ort

from dt_computer_vision.camera.types import Pixel
from duckietown_messages.actuators.differential_pwm import DifferentialPWM

from solution.config import (
    MODEL_PATH,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    STOP_DISTANCE,
    FORWARD_PWM,
)


class MLModel:
    def __init__(self):
        print("Initializing MLModel")

        self.ground_projector = None

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}"
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]

        self.input_name = inp.name
        self.in_dtype = (
            np.float16
            if inp.type == "tensor(float16)"
            else np.float32
        )

        self.net_h = int(inp.shape[2])
        self.net_w = int(inp.shape[3])

        print(f"ONNX INPUT SIZE: {self.net_w}x{self.net_h}")

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------
    def _preprocess(self, img_bgr):

        print("ORIGINAL IMAGE:", img_bgr.shape)

        self.orig_h = img_bgr.shape[0]
        self.orig_w = img_bgr.shape[1]

        # resize naar model input
        img = cv2.resize(img_bgr, (self.net_w, self.net_h))

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize
        img = img.astype(self.in_dtype) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # batch dimension
        img = np.expand_dims(img, axis=0)

        print("MODEL INPUT:", img.shape)
        print("INPUT RANGE:", img.min(), img.max())

        return img

    # -------------------------------------------------
    # POSTPROCESS
    # -------------------------------------------------
    def _postprocess(self, output):
        """
        Convert common YOLO ONNX output layouts to:
        [x1, y1, x2, y2, score, class_id]

        Ultralytics exports can return either NMS-ready detections
        (N, 6) or raw predictions shaped like (C, N)/(N, C).
        """

        out = np.asarray(output)
        print("POSTPROCESS INPUT SHAPE:", out.shape)

        # Remove batch dimension(s), but keep detection/channel dimensions.
        while out.ndim > 2 and out.shape[0] == 1:
            out = out[0]

        if out.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        if out.ndim != 2:
            print("Unsupported YOLO output rank:", out.ndim)
            return np.empty((0, 6), dtype=np.float32)

        # Already post-NMS: rows are [x1, y1, x2, y2, score, class].
        if out.shape[1] == 6:
            detections = out.astype(np.float32, copy=False)
            detections = detections[detections[:, 4] >= CONF_THRESHOLD]
            return detections[:MAX_DETECTIONS]

        # Some exports return [batch_id, x1, y1, x2, y2, score, class].
        if out.shape[1] == 7:
            detections = out[:, 1:7].astype(np.float32, copy=False)
            detections = detections[detections[:, 4] >= CONF_THRESHOLD]
            return detections[:MAX_DETECTIONS]

        # Raw YOLO output is commonly channels x anchors, e.g. 5 x 8400
        # for one class: cx, cy, w, h, class_score.
        if out.shape[0] >= 5 and (out.shape[0] < out.shape[1] or out.shape[1] < 5):
            out = out.T

        if out.shape[1] < 5:
            print("Unsupported YOLO output shape:", out.shape)
            return np.empty((0, 6), dtype=np.float32)

        boxes_cxcywh = out[:, :4].astype(np.float32, copy=False)
        class_scores = out[:, 4:].astype(np.float32, copy=False)

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = scores >= CONF_THRESHOLD

        if not np.any(keep):
            print("No raw YOLO predictions above threshold")
            return np.empty((0, 6), dtype=np.float32)

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep].astype(np.float32)

        x, y, w, h = boxes_cxcywh.T
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Map model-input coordinates back to the original camera image.
        scale_x = self.orig_w / float(self.net_w)
        scale_y = self.orig_h / float(self.net_h)
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, self.orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, self.orig_h - 1)

        nms_boxes = []
        for bx1, by1, bx2, by2 in boxes_xyxy:
            nms_boxes.append([float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.astype(float).tolist(),
            CONF_THRESHOLD,
            IOU_THRESHOLD,
        )

        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)

        indices = np.asarray(indices).reshape(-1)[:MAX_DETECTIONS]

        detections = np.column_stack(
            [
                boxes_xyxy[indices],
                scores[indices],
                class_ids[indices],
            ]
        )

        return detections.astype(np.float32, copy=False)

    # -------------------------------------------------
    # INFERENCE
    # -------------------------------------------------
    def _run_detector(self, img_bgr):

        x = self._preprocess(img_bgr)

        outputs = self.session.run(
            None,
            {self.input_name: x}
        )

        print("NUM OUTPUTS:", len(outputs))

        for i, o in enumerate(outputs):
            print(f"OUTPUT {i} SHAPE:", o.shape)

        out = outputs[0]

        print("RAW OUTPUT SAMPLE:")
        print(out.reshape(-1)[:10])

        detections = self._postprocess(out)
        print("POSTPROCESSED DETECTIONS:", detections.shape)

        return detections

    # -------------------------------------------------
    # STOP LOGIC
    # -------------------------------------------------
    def _should_stop(self, detections):

        stop = False

        print("TOTAL DETECTIONS:", len(detections))

        for det in detections:

            if len(det) < 6:
                continue

            x1, y1, x2, y2, score, cls = det[:6]

            print(
                f"Detection: "
                f"{x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f}, "
                f"conf={score:.3f}"
            )

            # confidence filtering
            if score < CONF_THRESHOLD:
                continue

            print("VALID DETECTION FOUND")

            # tijdelijke test:
            # zodra model iets detecteert -> stoppen
            stop = True

            # later kan je distance logic toevoegen

        return stop

    # -------------------------------------------------
    # GROUND PROJECTOR
    # -------------------------------------------------
    def set_ground_projector(self, gp):
        self.ground_projector = gp

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------
    def get_wheel_velocities_from_image(self, img):

        try:
            detections = self._run_detector(img)

        except Exception as e:

            print("ONNX inference error:", e)

            return [
                DifferentialPWM(left=0.0, right=0.0),
                None,
            ]

        should_stop = self._should_stop(detections)

        if should_stop:

            print("STOPPING")

            return [
                DifferentialPWM(left=0.0, right=0.0),
                detections,
            ]

        print("DRIVING")

        return [
            DifferentialPWM(
                left=FORWARD_PWM,
                right=FORWARD_PWM,
            ),
            detections,
        ]
