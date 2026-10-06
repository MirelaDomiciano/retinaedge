"""Fake LiteRT interpreter + a fake deploy package, so the app is testable without the real models."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZE = 64


class FakeInterpreter:
    """Mimics the LiteRT Interpreter API. Output = fixed probabilities; records the last input tensor."""

    def __init__(self, dtype=np.uint8, quant=(1.0, 0), probs=(0.05, 0.05, 0.8, 0.05, 0.05), shape=(1, SIZE, SIZE, 3)):
        self.dtype, self.quant, self.probs, self.shape = dtype, quant, np.array(probs, np.float32), shape
        self.last_input = None

    def allocate_tensors(self): pass
    def get_input_details(self): return [{"index": 0, "shape": np.array(self.shape), "dtype": self.dtype, "quantization": self.quant}]
    def get_output_details(self): return [{"index": 1}]
    def set_tensor(self, index, value):
        assert value.dtype == self.dtype and value.shape == self.shape
        self.last_input = value
    def invoke(self): pass
    def get_tensor(self, index): return self.probs[np.newaxis]


def fundus(size=256, brightness=150, cut=False):
    """Synthetic fundus photo: orange disc on black (optionally cut at top/bottom like many APTOS images)."""
    im = Image.new("RGB", (size, size), (0, 0, 0))
    ImageDraw.Draw(im).ellipse((20, 20, size - 20, size - 20), fill=(brightness, int(brightness * .5), int(brightness * .2)))
    return im.crop((0, 50, size, size - 50)) if cut else im


@pytest.fixture
def package(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "samples").mkdir()
    models = []
    for name, variant in [("mobilenetv2_64_fp16", "fp16"), ("mobilenetv2_64_int8_c300", "int8_c300")]:
        (tmp_path / "models" / f"{name}.tflite").write_bytes(b"x" * 1000)
        models.append({"name": name, "file": f"models/{name}.tflite", "variant": variant, "backbone": "mobilenetv2",
                       "qwk_thresholds": [0.5, 1.5, 2.5, 3.5], "referable_threshold": 0.5,
                       "test": {"QWK (thresholds)": 0.9}})
    cfg = {"img_size": SIZE, "grade_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
           "referable_min_grade": 2, "default_model": "mobilenetv2_64_fp16", "models": models}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    for g in range(5):
        fundus().save(tmp_path / "samples" / f"grade{g}_img{g}.png")
    return tmp_path


@pytest.fixture
def factory():
    """interpreter_factory that returns a FakeInterpreter; tests can tweak .kwargs before loading."""
    class F:
        kwargs = {}
        last = None
        def __call__(self, path, threads):
            assert Path(path).exists()
            F.last = FakeInterpreter(**self.kwargs)
            return F.last
    return F()
