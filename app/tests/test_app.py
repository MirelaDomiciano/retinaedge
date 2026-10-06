import io
import json

import pytest
from PIL import Image

import app as appmod
from engine import Engine
from conftest import fundus


@pytest.fixture
def client(package, factory, monkeypatch):
    monkeypatch.setattr(appmod, "SAMPLES", package / "samples")
    appmod.ENGINE = Engine(package, interpreter_factory=factory)
    appmod.CAMERA = None
    appmod.STATE.update(live=False, last=None, preview_jpeg=None, error=None)
    return appmod.app.test_client()


def png_bytes(im):
    buf = io.BytesIO(); im.save(buf, "PNG"); buf.seek(0); return buf


def test_page_and_status(client):
    assert client.get("/").status_code == 200
    s = client.get("/api/status").get_json()
    assert s["model"] == "mobilenetv2_64_fp16" and "cpu_percent" in s["system"]


def test_upload_classifies(client):
    r = client.post("/api/classify", data={"image": (png_bytes(fundus()), "f.png")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["result"]["status"] == "ok" and j["preview"].startswith("data:image/jpeg")


def test_upload_garbage_is_400(client):
    r = client.post("/api/classify", data={"image": (io.BytesIO(b"not an image"), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "readable" in r.get_json()["error"]


def test_upload_missing_file_is_400(client):
    assert client.post("/api/classify").status_code == 400


def test_samples_listed_and_classified(client):
    names = client.get("/api/samples").get_json()
    assert len(names) == 5
    assert client.post(f"/api/sample/{names[0]}").get_json()["result"]["status"] == "ok"


def test_sample_path_traversal_blocked(client):
    assert client.post("/api/sample/..%2Fconfig.json").status_code == 404


def test_model_switch(client):
    r = client.post("/api/model", json={"key": "mobilenetv2_64_int8_c300"})
    assert r.status_code == 200 and appmod.ENGINE.model_key == "mobilenetv2_64_int8_c300"
    assert client.post("/api/model", json={"key": "nope"}).status_code == 400


def test_live_without_camera_is_409(client):
    assert client.post("/api/live", json={"on": True}).status_code == 409


def test_benchmark_writes_results(package, factory, monkeypatch):
    monkeypatch.setattr(appmod, "SAMPLES", package / "samples")
    rows = appmod.benchmark(Engine(package, interpreter_factory=factory), runs=2, out_dir=package)
    assert len(rows) == 2 and rows[0]["n_inferences"] == 10          # 5 samples x 2 runs, warm-up excluded
    assert rows[0]["sample_accuracy"] == 0.2                          # fake model always says grade 2
    assert json.loads((package / "benchmark_results.json").read_text())[1]["model"] == "mobilenetv2_64_int8_c300"


def test_true_grade_from_filename():
    assert appmod.true_grade("grade3_abc.png") == 3 and appmod.true_grade("abc.png") is None


def test_live_loop_classifies_camera_frames(client):
    """Fake camera -> live mode on -> the background loop produces results and an FPS figure."""
    import threading, time
    import numpy as np

    class FakeCam:
        available, backend, error = True, "fake", None
        def latest(self): return np.asarray(fundus())
    appmod.CAMERA = FakeCam()
    assert client.post("/api/live", json={"on": True}).status_code == 200
    threading.Thread(target=appmod.live_loop, daemon=True).start()
    time.sleep(0.6)
    s = client.get("/api/status").get_json()
    appmod.STATE["live"] = False
    assert s["live"] and s["last"]["status"] == "ok" and s["fps"] > 0
    assert client.get("/preview.jpg").status_code == 200
