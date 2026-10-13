# Results

Raspberry Pi 5 benchmark, generated on the Pi with `python app.py --benchmark` and copied from
`app/benchmark_results.*`.

| File | Content |
|---|---|
| `benchmark_results.csv` | Latency (mean, p50, p95), pre-processing, end-to-end time, FPS, CPU, model RAM and temperature per model |
| `benchmark_results.json` | Same data as JSON |

The figures and CSVs from training and evaluation (Colab) are in [`evidence/`](../evidence):

| Folder | Content |
|---|---|
| `dataset/` | Duplicate and conflicting pairs, class distribution |
| `training/` | Training curves |
| `evaluation/` | Keras models on the test set: summary, confusion matrix, ROC |
| `quantization/` | TFLite variants on validation and test, QWK per variant, ROC after quantization |
| `raspi_test/` | Screenshots of the live app on the Pi |
