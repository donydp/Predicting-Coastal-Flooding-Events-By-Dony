# HDR Challenge – Coastal Flood Prediction (XGBoost)

This repository contains my training code and Codabench submission code for the **HDR challenge** (coastal flooding prediction from tide-gauge sea level time series).

Contents:
- `training/train_submit_tuned.py` – training pipeline (random search over XGBoost hyperparameters + top-K ensemble + calibration shift). Produces `model.pkl`.
- `submission/model.py` – inference code used by Codabench (robust to Final Phase input packaging differences: CSV **or** MAT inputs).
- `submission/model.pkl` – exported inference-only model payload (portable: stores XGBoost Booster models as JSON strings, plus metadata).
- `submission/requirements.txt` – minimal inference dependencies.
- `submission/make_submission_zip.py` – utility to create a Codabench-ready `submission.zip`.

> Notes
> - The exported `model.pkl` does **not** pickle sklearn model objects (avoids version mismatch issues). It stores only JSON strings for XGBoost boosters and simple python/numpy metadata.
> - The inference script is designed to work when Codabench provides either `test_hourly.csv`/`test_index.csv` **or** only MAT files (Final Phase behavior).

---

## 1) Setup

Create a Python environment (Python 3.10+ recommended):

```bash
pip install -r submission/requirements.txt
```

For training, you will also need `scipy` (already included) and enough RAM/CPU (GPU optional).

---

## 2) Training

### Option A: Train from MAT

```bash
python training/train_submit_tuned.py \
  --mat NEUSTG_19502020_12stations.mat \
  --threshold_mat Seed_Coastal_Stations_Thresholds.mat \
  --out_dir submission \
  --trials 80 \
  --keep 5
```

### Option B: Train from CSV

```bash
python training/train_submit_tuned.py \
  --train_csv train_hourly.csv \
  --threshold_mat Seed_Coastal_Stations_Thresholds.mat \
  --out_dir submission \
  --trials 80 \
  --keep 5
```

### GPU training (optional)

```bash
python training/train_submit_tuned.py --train_csv train_hourly.csv --use_gpu --trials 80 --keep 5 --out_dir submission
```

This writes:
- `submission/model.pkl`

---

## 3) Create Codabench submission zip

```bash
python submission/make_submission_zip.py --out submission.zip
```

Upload `submission.zip` to Codabench.

---

## 4) Inference behavior

`submission/model.py` will:
1. Prefer CSV inputs if they exist:
   - `test_hourly.csv`
   - `test_index.csv`
2. Otherwise, it searches the input directory for `.mat` files and loads tide-gauge data from MAT.
3. It writes:
   - `predictions.csv` with columns `id,y_prob`

---

## Reproducibility & environment notes

- The inference payload stores XGBoost boosters as JSON strings for portability.
- For best reproducibility, use a fixed seed (`--seed 42` default).

---

## License

Code in this repository is provided for challenge reproducibility and review.
