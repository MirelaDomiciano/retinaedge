# Training (Google Colab)

- `RetinaEdge_training.ipynb`: full pipeline (APTOS download, deduplication and split, preprocessing,
  ResNet-50 / MobileNetV2 training at 224 and 384 px, evaluation with validation-fitted thresholds,
  TFLite conversion and quantization study, deploy package).
- `dedup_split.py`: MD5 + perceptual-hash deduplication and stratified split (loaded with `%run`).

The notebook loads `preprocess.py` and `dedup_split.py` from Google Drive
(`/content/drive/MyDrive/tcc/scripts/`). The canonical `preprocess.py` is `app/preprocess.py`;
the Drive copy must be identical (the Deploy Package cell checks pixel parity).
