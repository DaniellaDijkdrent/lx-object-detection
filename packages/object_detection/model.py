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
        print(out[0][:5])

        return out[0]

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