"""
RetinaEdge inference engine: config.json + .tflite + preprocess.py -> grade, referable flag, warnings.

Pure logic (no Flask, no camera) so it can be unit-tested on any machine.
The input handling mirrors tflite_predict() in the training notebook:
the normalization is INSIDE the model, so raw pixels go in.
  - uint8 input  -> raw pixels, quantized with the tensor's (scale, zero_point) (identity for our models)
  - float32 input -> raw pixels as float, NOT divided by 255
"""
import json
import time
from pathlib import Path

import numpy as np

import preprocess as pp

GRADES = np.arange(5)

# Error-case thresholds (documented in the report; tuned on the bundled samples)
DARK_MEAN = 25           # mean gray level of the central region below this = image too dark
CORNER_DARK = 40         # a corner patch darker than this counts as the black fundus background
LOW_CONFIDENCE = 0.45    # highest class probability below this = low confidence


def looks_like_fundus(image):
    """A fundus photo is a bright disc on black: inside the retina box (before the black padding),
    at least 2 of the 4 corners are dark. Catches camera frames where 'something bright' was found
    but it is not a retina (lit room, screen bezel in the frame)."""
    im = image.convert("RGB")
    box = pp.find_retina_bbox(im)
    if box is None:
        return False
    im = im.crop(box)
    im.thumbnail((128, 128))
    gray = np.asarray(im, dtype=np.float32).mean(axis=2)
    k = max(2, min(gray.shape) // 12)
    corners = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
    return sum(c.mean() < CORNER_DARK for c in corners) >= 2


def central_brightness(arr):
    """Mean gray level of the central half of the image (always retina, never the black border)."""
    h, w = arr.shape[:2]
    return float(arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean())


def _load_interpreter(path, num_threads):
    """LiteRT runtime (same as the notebook); falls back to tflite_runtime / tensorflow if needed."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    return Interpreter(model_path=str(path), num_threads=num_threads)


def apply_thresholds(probs, thresholds):
    """Expected grade (probability-weighted mean of 0..4) cut by the validation thresholds."""
    return int(np.digitize(float(probs @ GRADES), np.sort(thresholds)))


class Engine:
    def __init__(self, package_dir, model_key=None, num_threads=4, interpreter_factory=_load_interpreter):
        self.dir = Path(package_dir)
        self.config = json.loads((self.dir / "config.json").read_text())
        self.size = int(self.config["img_size"])
        self.grade_names = self.config["grade_names"]
        self.referable_min = int(self.config.get("referable_min_grade", 2))
        self.num_threads = num_threads
        self._factory = interpreter_factory
        self.model_key = None
        self.load(model_key or self.config["default_model"])

    @property
    def models(self):
        """config.json lists models; index them by name (e.g. 'mobilenetv2_384_fp16')."""
        return {m["name"]: m for m in self.config["models"]}

    def load(self, key):
        """Loads (or switches to) a model listed in config.json. Raises KeyError / FileNotFoundError clearly."""
        if key not in self.models:
            raise KeyError(f"unknown model '{key}'; available: {', '.join(self.models)}")
        spec = self.models[key]
        path = self.dir / spec["file"]          # "models/<name>.tflite"
        if not path.exists():
            raise FileNotFoundError(f"model file missing: {path}")
        it = self._factory(path, self.num_threads)
        it.allocate_tensors()
        self.interpreter, self.spec, self.model_key = it, spec, key
        self.inp = it.get_input_details()[0]
        self.out = it.get_output_details()[0]
        expected = (1, self.size, self.size, 3)
        if tuple(self.inp["shape"]) != expected:
            raise ValueError(f"{key}: model input {tuple(self.inp['shape'])} != {expected} from config")
        self.size_mb = path.stat().st_size / 1e6

    def _to_input(self, arr):
        x = arr[np.newaxis]
        if self.inp["dtype"] == np.uint8:
            scale, zero = self.inp["quantization"]
            if scale:  # (0, 0) means "not quantized" -> pass raw pixels
                x = np.clip(np.round(x / scale + zero), 0, 255)
            return x.astype(np.uint8)
        return x.astype(np.float32)   # raw 0-255 floats: no /255, normalization is inside the model

    def predict_probs(self, arr):
        """uint8 array (size, size, 3) -> probabilities (5,), and inference time in ms."""
        self.interpreter.set_tensor(self.inp["index"], self._to_input(arr))
        t0 = time.perf_counter()
        self.interpreter.invoke()
        ms = (time.perf_counter() - t0) * 1000
        probs = np.asarray(self.interpreter.get_tensor(self.out["index"])[0], dtype=np.float64)
        return probs, ms

    def classify(self, image):
        """PIL image (camera frame or file) -> result dict. Never raises for bad images: returns status + message."""
        t0 = time.perf_counter()
        im, found = pp.preprocess_image(image, self.size)
        pre_ms = (time.perf_counter() - t0) * 1000
        arr = pp.to_array(im)
        base = {"model": self.model_key, "preprocess_ms": round(pre_ms, 1), "preview": im}

        if not found or not looks_like_fundus(image):
            return {**base, "status": "no_retina",
                    "message": "No retina found. Center the fundus image in the frame."}
        brightness = central_brightness(arr)
        if brightness < DARK_MEAN:
            return {**base, "status": "too_dark", "brightness": round(brightness, 1),
                    "message": "Image too dark. Increase screen brightness or reduce glare."}

        probs, inf_ms = self.predict_probs(arr)
        grade = apply_thresholds(probs, self.spec["qwk_thresholds"])
        p_ref = float(probs[self.referable_min:].sum())
        result = {**base, "status": "ok", "grade": grade, "grade_name": self.grade_names[grade],
                  "argmax_grade": int(probs.argmax()), "probs": [round(float(p), 4) for p in probs],
                  "p_referable": round(p_ref, 4), "referable": p_ref >= self.spec["referable_threshold"],
                  "confidence": round(float(probs.max()), 4), "inference_ms": round(inf_ms, 1),
                  "brightness": round(brightness, 1)}
        if probs.max() < LOW_CONFIDENCE:
            result["status"] = "low_confidence"
            result["message"] = "Low confidence: grades are close. Recapture or refer to a specialist."
        return result
