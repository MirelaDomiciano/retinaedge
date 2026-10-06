# RetinaEdge — Raspberry Pi app

Diabetic retinopathy grading (ICDR 0–4 + referable DR) on a Raspberry Pi 5 with LiteRT.
Research prototype, not a medical device.

## Files

`config.json`, `models/` and `samples/` are not in the repository (size and dataset licence).
Unzip the deploy package (GitHub Release or the `Deploy Package` notebook cell) into `app/`,
next to these files:

| File | Role |
|---|---|
| `app.py` | Flask web app, camera thread, live inference loop, benchmark mode |
| `engine.py` | Inference: preprocessing, LiteRT, thresholds, error cases (no Flask, testable) |
| `requirements.txt` | Python dependencies |
| `tests/` | Unit tests (fake interpreter, run anywhere) |

## Install (Raspberry Pi OS Bookworm 64-bit)

```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-venv
git clone <repo-url> && cd retinaedge/app
unzip -n ../retinaedge_384.zip -d .   # adds config.json, models/, samples/; keeps the repo's .py files
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests        # all tests should pass
```

## Run

```bash
python app.py                     # open http://<pi-ip>:5000 on any device in the network
python app.py --no-camera         # upload / sample mode only
python app.py --model mobilenetv2_384_int8_c300
python app.py --benchmark --runs 5   # every model in config.json -> benchmark_results.csv/json
```

The camera mode re-captures fundus photographs shown on a screen (a simulation of a fundus
camera). Dim the room lights and fill the frame with the retina.

## How a frame is classified

1. `preprocess.py` (same code as training): crop the retina, pad to square, resize to 384 px.
2. Checks before inference: no retina found / not a fundus image, image too dark.
3. Raw `uint8` pixels go to the model (normalization is inside the model; nothing is divided by 255).
4. Grade = expected grade (probabilities · [0..4]) cut by the 4 thresholds fitted on validation.
   Referable = P(2)+P(3)+P(4) ≥ the validation cut-off. Highest probability < 0.45 is flagged as low confidence.

## Benchmark protocol

The first inference after loading each model is discarded (warm-up). Latency is reported as mean,
median and p95 of `invoke()`, plus preprocessing and end-to-end time, FPS, CPU %, model RAM and
CPU temperature. `sample_accuracy` uses the bundled samples (2 per grade): a sanity check, not a metric.
