#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_submit_tuned.py

Fine-tuning XGBoost (random search) + top-K ensemble + calibration-shift.
Output model.pkl is inference-only and portable (stores Booster JSON strings).

Usage:
  python train_submit_tuned.py --mat NEUSTG_19502020_12stations.mat --out_dir .
or:
  python train_submit_tuned.py --train_csv train_hourly.csv --out_dir .

Recommended (GPU if available):
  python train_submit_tuned.py --train_csv train_hourly.csv --use_gpu --trials 80 --keep 5 --out_dir .

Outputs:
  out_dir/model.pkl
"""

import os
import argparse
import time
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.io import loadmat

# XGBoost required for this tuned pipeline
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    xgb = None


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_HIST_DAYS = 7
DEFAULT_FUTURE_DAYS = 14
RANDOM_SEED = 42


# -----------------------------
# Utilities
# -----------------------------
def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=float(matlab_datenum) % 1) - timedelta(days=366)

def load_mat_hourly(mat_path):
    mat = loadmat(mat_path)
    lat = mat["lattg"].flatten()
    lon = mat["lontg"].flatten()
    sea_level = mat["sltg"]  # (time, stations)
    station_names = [s[0] for s in mat["sname"].flatten()]
    t = mat["t"].flatten()
    time_dt = pd.to_datetime([matlab2datetime(v) for v in t])

    dfs = []
    for j, s in enumerate(station_names):
        arr = sea_level[:, j]
        dfj = pd.DataFrame(
            {
                "time": time_dt,
                "station_name": s,
                "latitude": float(lat[j]),
                "longitude": float(lon[j]),
                "sea_level": arr,
            }
        )
        dfs.append(dfj)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(["station_name", "time"], inplace=True)

    # robust fill
    df["sea_level"] = df.groupby("station_name")["sea_level"].transform(lambda x: x.ffill().bfill())
    if df["sea_level"].isna().any():
        df["sea_level"] = df["sea_level"].fillna(df["sea_level"].median())
    return df

def _try_extract_thresholds_from_mat(thr_mat_path):
    """
    Best-effort parser for Seed_Coastal_Stations_Thresholds.mat (variable names unknown).
    Returns dict station_name -> threshold_float
    """
    if not os.path.exists(thr_mat_path):
        return {}

    mat = loadmat(thr_mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    station_candidates = []
    thr_candidates = []
    for k in keys:
        lk = k.lower()
        v = mat[k]
        if ("name" in lk) or ("station" in lk) or ("sname" in lk):
            station_candidates.append((k, v))
        if ("thr" in lk) or ("thres" in lk) or ("threshold" in lk):
            thr_candidates.append((k, v))

    def normalize_station_names(arr):
        try:
            flat = arr.flatten()
        except Exception:
            return []
        out = []
        for x in flat:
            try:
                if isinstance(x, np.ndarray):
                    # common: array(['ABC'], dtype='<U3') or char array
                    if x.dtype.kind in ("U", "S"):
                        out.append(str(x.squeeze()))
                    else:
                        # maybe char codes
                        try:
                            out.append("".join([chr(int(c)) for c in x.squeeze()]))
                        except Exception:
                            out.append(str(x.squeeze()))
                else:
                    out.append(str(x))
            except Exception:
                out.append(str(x))
        return [s.strip() for s in out]

    def normalize_thresholds(arr):
        try:
            a = np.array(arr).astype(float).squeeze()
            if a.ndim == 0:
                return [float(a)]
            return [float(v) for v in a.flatten()]
        except Exception:
            return []

    for sk, sv in station_candidates:
        st_names = normalize_station_names(sv)
        if not st_names:
            continue
        for tk, tv in thr_candidates:
            thrs = normalize_thresholds(tv)
            if len(thrs) == len(st_names) and len(thrs) > 0:
                return {st_names[i]: float(thrs[i]) for i in range(len(st_names))}
    return {}

def hourly_to_daily(df_hourly):
    df = df_hourly.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")

    daily = (
        df.groupby(["station_name", "date"])
        .agg(
            sea_level=("sea_level", "mean"),
            sea_level_max=("sea_level", "max"),
            sea_level_min=("sea_level", "min"),
            sea_level_std=("sea_level", "std"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .reset_index()
    )
    daily["sea_level_std"] = daily["sea_level_std"].fillna(0.0)

    daily.sort_values(["station_name", "date"], inplace=True)
    daily.reset_index(drop=True, inplace=True)

    g = daily.groupby("station_name")["sea_level"]

    # rolling means/stds
    daily["sea_level_3d_mean"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily["sea_level_7d_mean"] = g.transform(lambda x: x.rolling(7, min_periods=1).mean())
    daily["sea_level_14d_mean"] = g.transform(lambda x: x.rolling(14, min_periods=1).mean())

    daily["sea_level_3d_std"] = g.transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0.0)
    daily["sea_level_14d_std"] = g.transform(lambda x: x.rolling(14, min_periods=1).std()).fillna(0.0)

    daily["sea_level_14d_max"] = g.transform(lambda x: x.rolling(14, min_periods=1).max())
    daily["sea_level_14d_min"] = g.transform(lambda x: x.rolling(14, min_periods=1).min())
    daily["sea_level_14d_range"] = daily["sea_level_14d_max"] - daily["sea_level_14d_min"]

    # trend proxy
    daily["sea_level_trend7"] = g.transform(lambda x: (x - x.shift(6)) / 6.0).fillna(0.0)

    # diffs
    daily["sea_level_diff1"] = g.transform(lambda x: x.diff()).fillna(0.0)
    daily["sea_level_diff7"] = g.transform(lambda x: x.diff(7)).fillna(0.0)

    # seasonality
    dt = pd.to_datetime(daily["date"])
    daily["doy"] = dt.dt.dayofyear.astype(int)
    daily["doy_sin"] = np.sin(2.0 * np.pi * daily["doy"] / 365.25)
    daily["doy_cos"] = np.cos(2.0 * np.pi * daily["doy"] / 365.25)
    daily["month"] = dt.dt.month.astype(int)
    daily["weekday"] = dt.dt.weekday.astype(int)

    # station stats for anomaly
    stats = (
        daily.groupby("station_name")["sea_level"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "st_mean", "std": "st_std"})
        .reset_index()
    )
    stats["st_std"] = stats["st_std"].fillna(0.0)
    daily = daily.merge(stats, on="station_name", how="left")
    daily["sea_level_anom"] = (daily["sea_level"] - daily["st_mean"]) / (daily["st_std"] + 1e-9)

    daily.sort_values(["station_name", "date"], inplace=True)
    daily.reset_index(drop=True, inplace=True)
    return daily

def add_station_onehot(daily, station_names):
    st_to_idx = {s: i for i, s in enumerate(station_names)}
    idx = daily["station_name"].map(st_to_idx).fillna(-1).astype(int).values
    ohe = np.zeros((len(daily), len(station_names)), dtype=np.float32)
    ok = idx >= 0
    ohe[np.where(ok)[0], idx[ok]] = 1.0
    return ohe

def build_windows_and_labels(daily, hist_days, future_days, threshold_map=None):
    if threshold_map is None:
        threshold_map = {}

    # fallback threshold if official not found
    fallback_thr = (
        daily.groupby("station_name")["sea_level"]
        .agg(["mean", "std"])
        .assign(thr=lambda x: x["mean"] + 1.5 * x["std"])
        ["thr"]
        .to_dict()
    )

    thr_used = daily["station_name"].map(lambda s: threshold_map.get(s, fallback_thr.get(s, np.nan))).astype(float)
    thr_used = thr_used.fillna(daily["sea_level"].median())
    df = daily.copy()
    df["thr_used"] = thr_used.values
    df["flood"] = (df["sea_level_max"] > df["thr_used"]).astype(np.int8)

    per_day_feats = [
        "sea_level",
        "sea_level_max",
        "sea_level_min",
        "sea_level_std",
        "sea_level_3d_mean",
        "sea_level_7d_mean",
        "sea_level_14d_mean",
        "sea_level_3d_std",
        "sea_level_14d_std",
        "sea_level_14d_max",
        "sea_level_14d_min",
        "sea_level_14d_range",
        "sea_level_trend7",
        "sea_level_anom",
        "sea_level_diff1",
        "sea_level_diff7",
        "doy_sin",
        "doy_cos",
        "weekday",
        "month",
    ]

    station_names = sorted(df["station_name"].unique().tolist())
    ohe = add_station_onehot(df, station_names)

    X_list, y_list, meta_list = [], [], []

    for stn, grp in df.groupby("station_name"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < hist_days + future_days:
            continue

        grp_ohe = ohe[df["station_name"].values == stn]
        if len(grp_ohe) != len(grp):
            grp_ohe = np.zeros((len(grp), len(station_names)), dtype=np.float32)

        F = grp[per_day_feats].to_numpy(dtype=np.float32)
        st_mean = grp["st_mean"].to_numpy(dtype=np.float32)
        st_std = grp["st_std"].to_numpy(dtype=np.float32)
        lat = grp["latitude"].to_numpy(dtype=np.float32)
        lon = grp["longitude"].to_numpy(dtype=np.float32)
        flood = grp["flood"].to_numpy(dtype=np.int8)
        dates = pd.to_datetime(grp["date"]).to_numpy()

        max_i = len(grp) - hist_days - future_days + 1
        for i in range(max_i):
            hist_block = F[i:i + hist_days, :]
            if not np.isfinite(hist_block).all():
                continue

            x_flat = hist_block.reshape(-1)
            x_extra = np.concatenate(
                [
                    np.array([st_mean[i], st_std[i], lat[i], lon[i]], dtype=np.float32),
                    grp_ohe[i].astype(np.float32),
                ]
            )
            x = np.concatenate([x_flat, x_extra])

            fut = flood[i + hist_days:i + hist_days + future_days]
            y = 1 if np.max(fut) > 0 else 0

            X_list.append(x)
            y_list.append(y)
            meta_list.append(
                (stn,
                 pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                 pd.Timestamp(dates[i + hist_days]).strftime("%Y-%m-%d"))
            )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int8)
    meta_df = pd.DataFrame(meta_list, columns=["station", "hist_start", "future_start"])
    return X, y, meta_df, per_day_feats, station_names

def time_split_by_station(meta_df, val_frac=0.15):
    meta_df = meta_df.copy()
    meta_df["future_start_dt"] = pd.to_datetime(meta_df["future_start"])
    is_val = np.zeros(len(meta_df), dtype=bool)

    for stn, g in meta_df.groupby("station"):
        g = g.sort_values("future_start_dt")
        if len(g) < 10:
            continue
        cut = int(np.floor((1.0 - val_frac) * len(g)))
        is_val[g.index[cut:]] = True

    is_tr = ~is_val
    return is_tr, is_val

def f1_acc_mcc(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    if (2 * tp + fp + fn) > 0:
        f1 = (2.0 * tp) / float(2 * tp + fp + fn)
    else:
        f1 = 0.0

    acc = (tp + tn) / float(max(1, tp + tn + fp + fn))

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        mcc = 0.0
    else:
        mcc = ((tp * tn) - (fp * fn)) / float(np.sqrt(denom))

    return float(f1), float(acc), float(mcc)

def best_threshold_for_f1(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0
    best_acc = 0.0
    for thr in np.linspace(0.02, 0.98, 97):
        f1, acc, _ = f1_acc_mcc(y_true, y_prob, float(thr))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_acc = acc
    return best_thr, float(best_f1), float(best_acc)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def booster_to_json_str(bst):
    # robust serialization
    try:
        raw = bst.save_raw(raw_format="json")
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp = tf.name
        try:
            bst.save_model(tmp)
            with open(tmp, "rb") as f:
                b = f.read()
            return b.decode("utf-8", errors="ignore")
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass


# -----------------------------
# Fine tuning (random search)
# -----------------------------
def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def sample_params(rng):
    """
    Parameter space tuned for tabular time-window classification.
    """
    p = {}
    p["max_depth"] = int(rng.integers(3, 10))                 # 3..9
    p["min_child_weight"] = _loguniform(rng, 1.0, 30.0)      # 1..30
    p["subsample"] = float(rng.uniform(0.65, 1.0))           # 0.65..1.0
    p["colsample_bytree"] = float(rng.uniform(0.65, 1.0))    # 0.65..1.0
    p["eta"] = _loguniform(rng, 0.01, 0.10)                  # 0.01..0.1
    p["gamma"] = float(rng.uniform(0.0, 2.0))                # 0..2
    p["reg_lambda"] = _loguniform(rng, 0.5, 20.0)            # 0.5..20
    p["reg_alpha"] = _loguniform(rng, 1e-6, 2.0)             # ~0..2 (log)
    # stabilizer for imbalanced / noisy gradients
    p["max_delta_step"] = float(rng.uniform(0.0, 5.0))       # 0..5
    return p

def train_one_trial(X_tr, y_tr, X_va, y_va, seed, use_gpu):
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

    pos = max(1, int((y_tr == 1).sum()))
    neg = max(1, int((y_tr == 0).sum()))
    spw = float(neg / pos)

    tree_method = "gpu_hist" if use_gpu else "hist"

    # sample candidate params
    rng = np.random.default_rng(seed)
    hp = sample_params(rng)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": tree_method,
        "seed": int(seed),
        "verbosity": 0,
        "nthread": -1,
        "scale_pos_weight": spw,
        # practical stabilizers
        "max_bin": 256,
    }
    params.update(hp)

    # training control
    num_boost_round = 15000
    early_stopping_rounds = 350

    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    best_iter = int(getattr(bst, "best_iteration", 0)) + 1
    p_va = bst.predict(dva, iteration_range=(0, best_iter))

    f1_05, acc_05, mcc_05 = f1_acc_mcc(y_va, p_va, 0.5)
    thr_best, f1_best, acc_best = best_threshold_for_f1(y_va, p_va)

    # composite objective: prioritize F1, keep accuracy meaningful
    # (weights can be adjusted; this tends to preserve accuracy while pushing F1)
    score = float(f1_best + 0.03 * acc_best)

    return {
        "bst": bst,
        "params": params,
        "best_iter": best_iter,
        "score": score,
        "thr_best": thr_best,
        "f1_best": f1_best,
        "acc_best": acc_best,
        "f1_05": f1_05,
        "acc_05": acc_05,
        "mcc_05": mcc_05,
        "p_va": p_va,  # keep for ensemble threshold calibration
    }

def retrain_fixed_rounds(params, X_full, y_full, num_boost_round, use_gpu, seed):
    """
    Retrain on ALL data with fixed num_boost_round (from best_iter).
    This often gives a small bump vs keeping train/val-only model.
    """
    dtr = xgb.DMatrix(X_full, label=y_full)

    pos = max(1, int((y_full == 1).sum()))
    neg = max(1, int((y_full == 0).sum()))
    spw = float(neg / pos)

    p = dict(params)
    p["seed"] = int(seed)
    p["scale_pos_weight"] = spw
    p["tree_method"] = "gpu_hist" if use_gpu else "hist"
    p["verbosity"] = 0
    p["nthread"] = -1
    p["objective"] = "binary:logistic"
    p["eval_metric"] = "logloss"

    bst = xgb.train(
        params=p,
        dtrain=dtr,
        num_boost_round=int(num_boost_round),
        evals=[],
        verbose_eval=False,
    )
    return bst

def main(args):
    if not HAS_XGB:
        raise RuntimeError("xgboost is required. Install in your env: pip install xgboost")

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load hourly
    if args.train_csv and os.path.exists(args.train_csv):
        print("[train] Loading CSV:", args.train_csv, flush=True)
        df_hourly = pd.read_csv(args.train_csv, parse_dates=["time"])
    else:
        print("[train] Loading MAT:", args.mat, flush=True)
        df_hourly = load_mat_hourly(args.mat)

    # Load official thresholds if possible
    thr_map = {}
    if args.threshold_mat and os.path.exists(args.threshold_mat):
        thr_map = _try_extract_thresholds_from_mat(args.threshold_mat)
        print(f"[train] Threshold map loaded: {len(thr_map)} stations from {args.threshold_mat}", flush=True)
    else:
        print("[train] No threshold_mat found/used; fallback thr = mean+1.5*std", flush=True)

    # Daily + windows
    print("[train] Building daily features...", flush=True)
    daily = hourly_to_daily(df_hourly)

    print("[train] Building windows + labels...", flush=True)
    X, y, meta, per_day_feats, station_names = build_windows_and_labels(
        daily,
        hist_days=args.hist_days,
        future_days=args.future_days,
        threshold_map=thr_map,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No windows constructed. Check data coverage / hist_days / future_days")

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"[train] X={X.shape} pos={pos} neg={neg} pos_rate={pos/max(1,len(y)):.4f}", flush=True)

    # Time split
    is_tr, is_va = time_split_by_station(meta, val_frac=args.val_frac)
    X_tr, y_tr = X[is_tr], y[is_tr]
    X_va, y_va = X[is_va], y[is_va]
    print(f"[train] Split: train={len(y_tr)} val={len(y_va)}", flush=True)

    # Random search
    trials = int(args.trials)
    keep = int(args.keep)
    rng = np.random.default_rng(int(args.seed))

    print(f"[train] Random search trials={trials} keep={keep} use_gpu={args.use_gpu}", flush=True)

    best_list = []
    start = time.time()
    for t in range(trials):
        seed = int(args.seed + 1000 + t * 17 + int(rng.integers(0, 100000)))
        res = train_one_trial(X_tr, y_tr, X_va, y_va, seed=seed, use_gpu=args.use_gpu)

        best_list.append(res)
        # keep only top few in memory
        best_list.sort(key=lambda r: r["score"], reverse=True)
        best_list = best_list[: max(keep * 3, keep + 5)]

        if (t + 1) % max(1, trials // 10) == 0 or (t == 0):
            top = best_list[0]
            print(
                f"[trial {t+1:>4}/{trials}] "
                f"top score={top['score']:.5f} bestF1={top['f1_best']:.4f} bestThr={top['thr_best']:.3f} "
                f"F1@0.5={top['f1_05']:.4f} Acc@0.5={top['acc_05']:.4f} MCC@0.5={top['mcc_05']:.4f} "
                f"iter={top['best_iter']}",
                flush=True,
            )

    # Final selection
    best_list.sort(key=lambda r: r["score"], reverse=True)
    selected = best_list[:keep]
    print("[train] Selected top-K (bestF1):", [round(r["f1_best"], 4) for r in selected], flush=True)

    # Ensemble on VAL (from selected trial models)
    p_mat = np.column_stack([r["p_va"] for r in selected])
    p_ens = np.mean(p_mat, axis=1)

    f1_05, acc_05, mcc_05 = f1_acc_mcc(y_va, p_ens, 0.5)
    thr_best, f1_best, acc_best = best_threshold_for_f1(y_va, p_ens)

    # Calibration-shift: move logits so that thr_best becomes 0.5
    calib_bias = float(-logit(thr_best))
    p_adj = sigmoid(logit(p_ens) + calib_bias)
    f1_adj, acc_adj, mcc_adj = f1_acc_mcc(y_va, p_adj, 0.5)

    print(f"[train] Ensemble RAW @0.5:   F1={f1_05:.4f} Acc={acc_05:.4f} MCC={mcc_05:.4f}", flush=True)
    print(f"[train] Ensemble best_thr={thr_best:.3f} bestF1={f1_best:.4f} (acc@bestThr={acc_best:.4f})", flush=True)
    print(f"[train] Ensemble CAL @0.5:   F1={f1_adj:.4f} Acc={acc_adj:.4f} MCC={mcc_adj:.4f}", flush=True)

    # Retrain each selected config on FULL data (train+val)
    # Use its own best_iter as fixed rounds (common trick for small gain)
    print("[train] Retraining selected models on FULL data (fixed rounds)...", flush=True)
    X_full = X
    y_full = y

    final_boosters_json = []
    final_rounds = []
    for i, r in enumerate(selected):
        rounds = int(max(50, r["best_iter"]))
        # slight extension to exploit more data (safe)
        rounds = int(rounds * 1.05)

        seed = int(args.seed + 9999 + i * 101)
        bst_full = retrain_fixed_rounds(r["params"], X_full, y_full, rounds, args.use_gpu, seed=seed)
        final_boosters_json.append(booster_to_json_str(bst_full))
        final_rounds.append(rounds)
        print(f"  [full-train {i+1}/{keep}] rounds={rounds}", flush=True)

    payload = {
        "model_type": "xgb_randomsearch_ensemble_calibrated",
        "model_jsons": final_boosters_json,
        "hist_days": int(args.hist_days),
        "future_days": int(args.future_days),
        "per_day_features": per_day_feats,
        "station_names": station_names,
        "calib_bias": calib_bias,
        "meta": {
            "seed": int(args.seed),
            "trials": int(trials),
            "keep": int(keep),
            "val_frac": float(args.val_frac),
            "val_metrics_raw_at_0.5": {"f1": f1_05, "acc": acc_05, "mcc": mcc_05},
            "val_metrics_calib_at_0.5": {"f1": f1_adj, "acc": acc_adj, "mcc": mcc_adj},
            "thr_best_on_val": float(thr_best),
            "full_train_rounds": final_rounds,
        },
    }

    out_path = os.path.join(args.out_dir, "model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print("[train] Saved:", out_path, flush=True)
    print("[train] Total elapsed (s):", int(time.time() - t0), flush=True)
    print("[train] Done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="NEUSTG_19502020_12stations.mat")
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--threshold_mat", default="Seed_Coastal_Stations_Thresholds.mat")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--hist_days", type=int, default=DEFAULT_HIST_DAYS)
    ap.add_argument("--future_days", type=int, default=DEFAULT_FUTURE_DAYS)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args)
