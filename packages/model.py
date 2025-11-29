#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import onnxruntime as ort
from types import SimpleNamespace

from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from solution.config import MODEL_PATH, CONF_THRESHOLD, STOP_DISTANCE_M, FORWARD_PWM


class MLModel:
    def __init__(self):
        print("Initializing MLModel")
        self.model_path = MODEL_PATH
        self.conf_threshold = CONF_THRESHOLD
        self.stop_distance_m = STOP_DISTANCE_M
        self.forward_pwm = FORWARD_PWM
        self.ground_projector = None

        if not self.model_path.exists():
            raise FileNotFoundError("ONNX model not found:", self.model_path)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"], 
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]


    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]  # shape [1,N,6]
        return out[0]


    def _should_stop(self, detections: np.ndarray): 
        #TODO: review the logic!
        if detections.size == 0:
            return False

        for x1, y1, x2, y2, score, _ in detections:
            if score < self.conf_threshold:
                continue

            u_bottom = 0.5 * (x1 + x2)
            v_bottom = y2

            dist = self._estimate_distance(u_bottom, v_bottom)
            print(f"conf={score:.2f}  dist≈{dist:.3f} m")

            if dist < self.stop_distance_m:
                print("STOP: obstacle too close")
                return True

        return False


    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img


    def set_ground_projector(self, gp):
        self.ground_projector = gp


    def _estimate_distance(self, u_bottom, v_bottom):
        if self.ground_projector is None:
            raise ValueError(
                "Ground Projection is not set."
            )

        pix = SimpleNamespace(x=float(u_bottom), y=float(v_bottom))
        vec = self.ground_projector.camera.pixel2vector(pix)
        gp = self.ground_projector.vector2ground(vec)

        dist = float(np.linalg.norm([gp.x, gp.y]))
        return dist
        

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error {e}")
            return DifferentialPWM(left=0.0, right=0.0)

        stop = self._should_stop(detections)

        if stop:
            return DifferentialPWM(left=0.0, right=0.0)
        else:
            return DifferentialPWM(left=self.forward_pwm, right=self.forward_pwm)
