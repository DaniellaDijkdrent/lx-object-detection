#!/usr/bin/env python3

import numpy as np
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
            raise FileNotFoundError(f"ONNX model not found at: {MODEL_PATH}")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name

        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]

        print(f"Model loaded. Input shape: {self.net_h}x{self.net_w}")

    # -----------------------------
    # INFERENCE
    # -----------------------------
    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]

        print("Raw model output shape:", out.shape)

        print("MODEL RAW OUTPUT TYPE:", type(out))
    print("MODEL RAW OUTPUT SHAPE:", [o.shape for o in out] if isinstance(out, list) else out.shape)
    print("RAW OUTPUT SAMPLE:", out)

        return out[0]

    # -----------------------------
    # STOP LOGIC
    # -----------------------------
    def _should_stop(self, detections: np.ndarray):

        if detections is None or len(detections) == 0:
            return False

        stop = False

        for x1, y1, x2, y2, score, cls in detections:

            if score < CONF_THRESHOLD:
                continue

            print(f"Detection CONF={score:.2f} box=({x1},{y1},{x2},{y2})")

            # Convert center pixel to world position
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            pix = Pixel(x=cx, y=cy)

            try:
                vec = self.ground_projector.camera.pixel2vector(pix)
                ground_point = self.ground_projector.vector2ground(vec)

                distance = np.linalg.norm(ground_point)

                print(f"Distance to object: {distance:.3f} m")

                if distance < STOP_DISTANCE:
                    print("STOP: object too close!")
                    stop = True

            except Exception as e:
                print(f"Projection error: {e}")

                # fallback: if we can't estimate distance, assume danger
                stop = True

        return stop

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} != model input {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    # -----------------------------
    # EXTERNAL API
    # -----------------------------
    def set_ground_projector(self, gp):
        self.ground_projector = gp

    def get_wheel_velocities_from_image(self, img):

        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return [DifferentialPWM(0.0, 0.0), None]

        stop = self._should_stop(detections)

        if stop:
            return [DifferentialPWM(0.0, 0.0), detections]
        else:
            return [DifferentialPWM(FORWARD_PWM, FORWARD_PWM), detections]