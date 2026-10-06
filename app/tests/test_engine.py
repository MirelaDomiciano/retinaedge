import numpy as np
import pytest
from PIL import Image

import engine as eng
from engine import Engine, apply_thresholds
from conftest import SIZE, fundus


def test_loads_default_model_and_lists_models(package, factory):
    e = Engine(package, interpreter_factory=factory)
    assert e.model_key == "mobilenetv2_64_fp16"
    assert set(e.models) == {"mobilenetv2_64_fp16", "mobilenetv2_64_int8_c300"}


def test_switch_model(package, factory):
    e = Engine(package, interpreter_factory=factory)
    e.load("mobilenetv2_64_int8_c300")
    assert e.model_key == "mobilenetv2_64_int8_c300" and e.spec["variant"] == "int8_c300"


def test_unknown_model_raises_clear_error(package, factory):
    e = Engine(package, interpreter_factory=factory)
    with pytest.raises(KeyError, match="unknown model"):
        e.load("does_not_exist")
    assert e.model_key == "mobilenetv2_64_fp16"          # previous model still active


def test_missing_model_file(package, factory):
    (package / "models" / "mobilenetv2_64_int8_c300.tflite").unlink()
    e = Engine(package, interpreter_factory=factory)
    with pytest.raises(FileNotFoundError):
        e.load("mobilenetv2_64_int8_c300")


def test_input_shape_must_match_config(package, factory):
    factory.kwargs = {"shape": (1, 32, 32, 3)}
    with pytest.raises(ValueError, match="model input"):
        Engine(package, interpreter_factory=factory)


def test_uint8_input_is_raw_pixels(package, factory):
    """Normalization is inside the model: the uint8 tensor must equal the preprocessed pixels exactly."""
    factory.kwargs = {"dtype": np.uint8, "quant": (1.0, 0)}
    e = Engine(package, interpreter_factory=factory)
    arr = np.random.default_rng(0).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    e.predict_probs(arr)
    assert np.array_equal(factory.last.last_input[0], arr)


def test_float_input_is_not_divided_by_255(package, factory):
    factory.kwargs = {"dtype": np.float32, "quant": (0.0, 0)}
    e = Engine(package, interpreter_factory=factory)
    arr = np.full((SIZE, SIZE, 3), 200, np.uint8)
    e.predict_probs(arr)
    assert factory.last.last_input.max() == 200.0


def test_apply_thresholds_matches_notebook_rule():
    probs = np.array([0, 0, 1, 0, 0.0])                   # expected grade 2.0
    assert apply_thresholds(probs, [0.5, 1.5, 2.5, 3.5]) == 2
    assert apply_thresholds(probs, [0.5, 1.5, 1.9, 3.5]) == 3   # a lower cut moves it up
    assert apply_thresholds(np.array([.5, 0, 0, 0, .5]), [3.5, 0.5, 2.5, 1.5]) == 2   # thresholds sorted


def test_classify_ok_and_referable(package, factory):
    e = Engine(package, interpreter_factory=factory)
    r = e.classify(fundus())
    assert r["status"] == "ok" and r["grade"] == 2 and r["grade_name"] == "Moderate"
    assert r["referable"] and r["p_referable"] == pytest.approx(0.9, abs=1e-6)
    assert len(r["probs"]) == 5 and r["inference_ms"] >= 0


def test_classify_not_referable(package, factory):
    factory.kwargs = {"probs": (0.9, 0.05, 0.03, 0.01, 0.01)}
    r = Engine(package, interpreter_factory=factory).classify(fundus())
    assert r["grade"] == 0 and not r["referable"]


def test_cut_retina_still_classified(package, factory):
    assert Engine(package, interpreter_factory=factory).classify(fundus(cut=True))["status"] == "ok"


def test_black_image_reports_no_retina(package, factory):
    e = Engine(package, interpreter_factory=factory)
    r = e.classify(Image.new("RGB", (200, 200)))
    assert r["status"] == "no_retina" and "grade" not in r and factory.last.last_input is None


def test_bright_scene_without_fundus_reports_no_retina(package, factory):
    """A camera pointed at a lit room: 'something' is found, but the corners are not black."""
    scene = Image.fromarray(np.random.default_rng(1).integers(80, 255, (240, 320, 3), dtype=np.uint8))
    assert Engine(package, interpreter_factory=factory).classify(scene)["status"] == "no_retina"


def test_dim_fundus_reports_too_dark(package, factory):
    r = Engine(package, interpreter_factory=factory).classify(fundus(brightness=30))
    assert r["status"] == "too_dark" and "grade" not in r


def test_low_confidence_is_flagged_but_still_graded(package, factory):
    factory.kwargs = {"probs": (0.3, 0.3, 0.2, 0.1, 0.1)}
    r = Engine(package, interpreter_factory=factory).classify(fundus())
    assert r["status"] == "low_confidence" and "grade" in r and r["message"]


def test_classify_never_raises_on_odd_inputs(package, factory):
    e = Engine(package, interpreter_factory=factory)
    for im in [Image.new("L", (100, 100), 255), Image.new("RGBA", (50, 300)), Image.new("RGB", (1, 1))]:
        assert e.classify(im)["status"] in {"ok", "no_retina", "too_dark", "low_confidence"}
