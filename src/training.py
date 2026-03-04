"""Training pipeline for stress detection.

Usage (from project root):
    py -u src/training.py --approach all
    py -u src/training.py --approach loso --subjects 2 3 4
    py -u src/training.py --approach loso --device both --n-classes 3
"""
import argparse
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    WESAD_SUBJECTS, MODELS_DIR, RESULTS_DIR, RANDOM_STATE,
    WINDOW_SIZE, WINDOW_STEP, DEFAULT_MODELS, CV_FOLDS,
)
from wesad_loader import load_wesad
from preprocessing import preprocess_wesad_signal
from features import FeatureExtractor
from models import StressModel

logger = logging.getLogger(__name__)


# ── feature extraction per subject ────────────────────────────────────────────

def extract_subject_features(
    subject: Dict, window_sec: int = 60, step_sec: int = 30,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Preprocess signals → sliding-window features + majority-vote labels.

    Automatically handles wrist-only, chest-only, or combined (``both``)
    signal keys based on what is present in ``subject['signals']``.
    """
    signals = subject["signals"]
    sr_dict = subject["sampling_rates"]
    binary = subject["binary_labels"]
    valid_mask = subject["valid_mask"]

    # lowercase keys for preprocessing / feature extraction
    sig_lower = {}
    sr_lower = {}
    for key, arr in signals.items():
        sig_lower[key.lower()] = arr.astype(float)
        sr_lower[key.lower()] = sr_dict.get(key, 4)

    preprocessed = preprocess_wesad_signal(sig_lower, sr_lower, target_sr=4)

    # Build feature-extraction input: use preprocessed for low-rate signals,
    # raw for BVP/ECG/EMG/Resp (need native SR for peak detection).
    feat_sig: Dict[str, np.ndarray] = {}
    feat_sr: Dict[str, int] = {}

    _native_keys = {"bvp", "wrist_bvp", "ecg", "chest_ecg",
                    "emg", "chest_emg", "resp", "chest_resp"}

    for k in preprocessed:
        low = k.lower()
        if low in _native_keys:
            raw_key = next((rk for rk in sig_lower if rk.lower() == low), None)
            if raw_key:
                feat_sig[k] = sig_lower[raw_key]
                feat_sr[k] = sr_lower[raw_key]
        else:
            feat_sig[k] = preprocessed[k]
            feat_sr[k] = 4

    rows = FeatureExtractor.extract_windows(feat_sig, feat_sr,
                                            window_sec, step_sec)
    if not rows:
        return np.array([]), np.array([]), []

    # Assign label per window (majority vote among valid 4 Hz labels)
    labels_sr = 4
    y_list = []
    for r in rows:
        i0 = int(r["t0"] * labels_sr)
        i1 = int(r["t1"] * labels_sr)
        seg_lbl = binary[i0:i1]
        seg_val = valid_mask[i0:i1]
        if seg_val.sum() / max(len(seg_val), 1) >= 0.8:
            vals, cnts = np.unique(seg_lbl[seg_val], return_counts=True)
            y_list.append(vals[np.argmax(cnts)])
        else:
            y_list.append(-1)

    y = np.array(y_list)
    meta = {"window", "t0", "t1"}
    feat_names = [k for k in rows[0] if k not in meta]
    X = np.array([[r.get(f, np.nan) for f in feat_names] for r in rows])

    ok = y >= 0
    X, y = X[ok], y[ok]
    sid = subject["subject_id"]
    logger.info("  S%d: %d windows (classes: %s)",
                sid, len(X),
                {int(v): int((y == v).sum()) for v in np.unique(y)} if len(y) else {})
    return X, y, feat_names


# ── training approaches ──────────────────────────────────────────────────────

def train_subject_dependent(data: Dict, *, window_sec=60, step_sec=30,
                            cv_folds=CV_FOLDS) -> Dict:
    """Per-subject k-fold cross-validation."""
    results: Dict = {}
    for subj in data["subjects"]:
        sid = subj["subject_id"]
        X, y, _ = extract_subject_features(subj, window_sec, step_sec)
        if len(X) < cv_folds * 2:
            logger.warning("  S%d: too few windows (%d), skipped", sid, len(X))
            continue
        subj_res: Dict = {}
        for name, params in DEFAULT_MODELS.items():
            mdl = StressModel(name, params)
            try:
                cv = mdl.cross_validate(X, y, cv=cv_folds, verbose=False)
                subj_res[name] = {
                    "accuracy": float(cv["test_accuracy"].mean()),
                    "accuracy_std": float(cv["test_accuracy"].std()),
                    "f1": float(cv["test_f1"].mean()),
                    "f1_std": float(cv["test_f1"].std()),
                    "n_windows": len(X),
                }
            except Exception as e:
                subj_res[name] = {"error": str(e)}
        results[f"S{sid}"] = subj_res
    return results


def train_subject_independent(data: Dict, *, window_sec=60, step_sec=30,
                              test_ratio=0.2) -> Tuple[Dict, List[str]]:
    """Pool subjects, split by subject groups, train & evaluate."""
    all_X, all_y, all_sid = [], [], []
    feat_names = None
    for subj in data["subjects"]:
        X, y, fn = extract_subject_features(subj, window_sec, step_sec)
        if len(X):
            all_X.append(X); all_y.append(y)
            all_sid.extend([subj["subject_id"]] * len(X))
            feat_names = feat_names or fn

    if not all_X:
        return {}, []
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    sids = np.array(all_sid)

    # subject-wise split
    uniq = np.unique(sids)
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_ratio))
    test_set, train_set = set(uniq[:n_test]), set(uniq[n_test:])
    tr = np.isin(sids, list(train_set))
    te = np.isin(sids, list(test_set))
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]

    logger.info("Train subjects=%s (%d), Test subjects=%s (%d)",
                sorted(train_set), len(X_tr), sorted(test_set), len(X_te))

    results: Dict = {"train_subjects": sorted(train_set),
                     "test_subjects": sorted(test_set), "models": {}}
    for name, params in DEFAULT_MODELS.items():
        mdl = StressModel(name, params)
        try:
            mdl.fit(X_tr, y_tr, verbose=False)
            m = mdl.evaluate(X_te, y_te, verbose=False)
            mdl.save(str(MODELS_DIR / f"{name}_independent.joblib"))
            results["models"][name] = {
                k: float(m[k]) for k in ("accuracy", "f1", "precision", "recall")
            }
            results["models"][name]["roc_auc"] = float(m.get("roc_auc", np.nan))
            results["models"][name]["cm"] = m["confusion_matrix"].tolist()
        except Exception as e:
            results["models"][name] = {"error": str(e)}
    return results, feat_names or []


def train_loso(data: Dict, *, window_sec=60, step_sec=30) -> Dict:
    """Leave-One-Subject-Out cross-validation."""
    sf: Dict = {}
    for subj in data["subjects"]:
        X, y, _ = extract_subject_features(subj, window_sec, step_sec)
        if len(X):
            sf[subj["subject_id"]] = (X, y)

    sids = sorted(sf)
    logger.info("LOSO with %d subjects: %s", len(sids), sids)
    results = {n: [] for n in DEFAULT_MODELS}

    for test_sid in sids:
        X_te, y_te = sf[test_sid]
        X_tr = np.vstack([sf[s][0] for s in sids if s != test_sid])
        y_tr = np.concatenate([sf[s][1] for s in sids if s != test_sid])

        for name, params in DEFAULT_MODELS.items():
            mdl = StressModel(name, params)
            try:
                mdl.fit(X_tr, y_tr, verbose=False)
                m = mdl.evaluate(X_te, y_te, verbose=False)
                results[name].append({
                    "test_subject": test_sid,
                    "accuracy": float(m["accuracy"]),
                    "f1": float(m["f1"]),
                    "roc_auc": float(m.get("roc_auc", np.nan)),
                })
            except Exception as e:
                logger.error("LOSO S%d %s: %s", test_sid, name, e)

    summary: Dict = {}
    for name, folds in results.items():
        if folds:
            acc = [f["accuracy"] for f in folds]
            f1s = [f["f1"] for f in folds]
            summary[name] = {
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_std": float(np.std(acc)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "per_subject": folds,
            }
    return summary


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save(results: Dict, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RESULTS_DIR / f"{tag}_{ts}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results → %s", p)
    return p


def _print_table(results: Dict, approach: str):
    print(f"\n{'='*65}\n  {approach}\n{'='*65}")
    if approach == "subject_dependent":
        for subj, models in results.items():
            print(f"\n  {subj}:")
            for mn, m in models.items():
                if "error" in m:
                    print(f"    {mn:22s} ERROR: {m['error']}")
                else:
                    print(f"    {mn:22s} Acc={m['accuracy']:.4f}±{m['accuracy_std']:.4f}  "
                          f"F1={m['f1']:.4f}±{m['f1_std']:.4f}")
    elif approach == "loso":
        for mn, s in results.items():
            if "accuracy_mean" in s:
                print(f"  {mn:22s} Acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}  "
                      f"F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Stress detection training")
    ap.add_argument("--approach", default="subject_dependent",
                    choices=["subject_dependent", "subject_independent", "loso", "all"])
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--window", type=int, default=WINDOW_SIZE)
    ap.add_argument("--step", type=int, default=WINDOW_STEP)
    ap.add_argument("--device", default="wrist",
                    choices=["wrist", "chest", "both"],
                    help="Sensor device(s) to use")
    ap.add_argument("--n-classes", type=int, default=2, choices=[2, 3],
                    help="Number of classes: 2=binary, 3=3-class")
    ap.add_argument("--wesad-dir", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("Loading WESAD (device=%s, n_classes=%d) …",
                args.device, args.n_classes)
    data = load_wesad(subject_ids=args.subjects, wesad_dir=args.wesad_dir,
                      device=args.device, n_classes=args.n_classes)
    if not data["subjects"]:
        logger.error("No subjects loaded – check WESAD directory.")
        return

    approaches = (["subject_dependent", "subject_independent", "loso"]
                  if args.approach == "all" else [args.approach])
    w, s = args.window, args.step

    for approach in approaches:
        logger.info("\n%s  %s  %s", "#" * 20, approach, "#" * 20)
        if approach == "subject_dependent":
            r = train_subject_dependent(data, window_sec=w, step_sec=s)
            _print_table(r, approach); _save(r, approach)
        elif approach == "subject_independent":
            r, _ = train_subject_independent(data, window_sec=w, step_sec=s)
            _save(r, approach)
        elif approach == "loso":
            r = train_loso(data, window_sec=w, step_sec=s)
            _print_table(r, approach); _save(r, approach)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
