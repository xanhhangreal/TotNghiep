"""Deep-learning training pipeline for stress detection.

Mirrors the ML pipeline in ``training.py`` but uses PyTorch models from
``dl_models.py``.  Supports subject-dependent, subject-independent, and
LOSO evaluation with all WESAD modalities.

Usage (from project root):
    py -u src/dl_training.py --arch all --classes binary --approach loso
    py -u src/dl_training.py --arch cnn1d --classes both --approach all
    py -u src/dl_training.py --arch resnet1d --device both --approach loso
    py -u src/dl_training.py --arch all --classes both --approach loso --paper-protocol
"""
import argparse
import json
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import (
    WESAD_SUBJECTS, MODELS_DIR, RESULTS_DIR, RANDOM_STATE,
    WINDOW_SIZE, WINDOW_STEP, DL_BATCH_SIZE, DL_EPOCHS,
    DL_LEARNING_RATE, DL_WEIGHT_DECAY, DL_PATIENCE,
    DL_LR_PATIENCE, DL_MIN_LR, DL_DROPOUT, DL_MODELS,
    DEFAULT_MODELS, CV_FOLDS,
)
from wesad_loader import load_wesad
from preprocessing import preprocess_wesad_signal
from features import FeatureExtractor
from dl_models import build_dl_model, save_dl_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sanitize_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Convert NaN/Inf values to robust column medians."""
    X = np.asarray(X, dtype=float)
    bad = ~np.isfinite(X)
    if not bad.any():
        return X

    X = X.copy()
    col_med = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        fin = col[np.isfinite(col)]
        col_med[j] = float(np.median(fin)) if len(fin) else 0.0
    rr, cc = np.where(~np.isfinite(X))
    X[rr, cc] = col_med[cc]
    return X


def _split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split train/val while remaining robust on tiny or imbalanced sets."""
    if len(X) < 10 or val_ratio <= 0:
        return X, y, None, None

    unique, counts = np.unique(y, return_counts=True)
    stratify = y if len(unique) > 1 and counts.min() >= 2 else None
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=seed, stratify=stratify
        )
    except ValueError:
        return X, y, None, None

    if len(np.unique(y_tr)) < 2 or len(X_val) == 0:
        return X, y, None, None
    return X_tr, y_tr, X_val, y_val


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction (all modalities)
# ══════════════════════════════════════════════════════════════════════════════

def extract_subject_features(
    subject: Dict,
    window_sec: float = 60,
    step_sec: float = 30,
    device_mode: str = "both",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Preprocess signals → sliding-window features + majority-vote labels.

    Works with ``device='wrist'``, ``'chest'``, or ``'both'`` signals.
    """
    signals = subject["signals"]
    sr_dict = subject["sampling_rates"]
    labels_mapped = subject["binary_labels"]
    valid_mask = subject["valid_mask"]

    # lowercase keys for preprocessing / feature extraction
    sig_lower = {}
    sr_lower = {}
    for key, arr in signals.items():
        sig_lower[key.lower()] = arr.astype(float)
        sr_lower[key.lower()] = sr_dict.get(key, 4)

    preprocessed = preprocess_wesad_signal(sig_lower, sr_lower, target_sr=4)

    # Build feature-extraction input – use preprocessed for low-rate signals,
    # original for BVP/ECG (need native rate for peak-based HRV).
    feat_sig: Dict[str, np.ndarray] = {}
    feat_sr: Dict[str, int] = {}

    for k, v in preprocessed.items():
        low = k.lower()
        # For BVP and ECG use raw signal at native SR for accurate peaks
        if low in ("bvp", "wrist_bvp"):
            raw_key = next((rk for rk in sig_lower if rk.lower() == low), None)
            if raw_key:
                feat_sig[k] = sig_lower[raw_key]
                feat_sr[k] = sr_lower[raw_key]
            continue
        if low in ("ecg", "chest_ecg"):
            raw_key = next((rk for rk in sig_lower if rk.lower() == low), None)
            if raw_key:
                feat_sig[k] = sig_lower[raw_key]
                feat_sr[k] = sr_lower[raw_key]
            continue
        if low in ("emg", "chest_emg"):
            raw_key = next((rk for rk in sig_lower if rk.lower() == low), None)
            if raw_key:
                feat_sig[k] = sig_lower[raw_key]
                feat_sr[k] = sr_lower[raw_key]
            continue
        if low in ("resp", "chest_resp"):
            raw_key = next((rk for rk in sig_lower if rk.lower() == low), None)
            if raw_key:
                feat_sig[k] = sig_lower[raw_key]
                feat_sr[k] = sr_lower[raw_key]
            continue
        # Preprocessed (resampled to 4 Hz) for EDA, TEMP, ACC
        feat_sig[k] = v
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
        seg_lbl = labels_mapped[i0:i1]
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
    X = _sanitize_feature_matrix(X)

    ok = y >= 0
    X, y = X[ok], y[ok]
    sid = subject["subject_id"]
    n_classes = len(np.unique(y[y >= 0])) if len(y) else 0
    logger.info("  S%d: %d windows, %d features, %d classes",
                sid, len(X), X.shape[1] if len(X) else 0, n_classes)
    return X, y, feat_names


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch helpers
# ══════════════════════════════════════════════════════════════════════════════

def _prep_tensors(X: np.ndarray, y: np.ndarray, scaler=None, fit=False):
    """Clean NaN, scale, return (X_tensor, y_tensor, scaler)."""
    X = np.asarray(X, dtype=float)
    X = X.copy()
    col_med = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        fin = col[np.isfinite(col)]
        col_med[j] = float(np.median(fin)) if len(fin) else 0.0
    rr, cc = np.where(~np.isfinite(X))
    X[rr, cc] = col_med[cc]

    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (B, 1, F)
    y_t = torch.tensor(y, dtype=torch.long)
    return X_t, y_t, scaler


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts.astype(float))
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop (single run)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    *,
    epochs: int = DL_EPOCHS,
    batch_size: int = DL_BATCH_SIZE,
    lr: float = DL_LEARNING_RATE,
    weight_decay: float = DL_WEIGHT_DECAY,
    patience: int = DL_PATIENCE,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """Train a PyTorch model with optional validation & early stopping.

    Returns dict with training history and best metrics.
    """
    model = model.to(DEVICE)

    # Prepare data
    scaler = StandardScaler()
    X_tr, y_tr, scaler = _prep_tensors(X_train.copy(), y_train.copy(),
                                        scaler, fit=True)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        assert X_val is not None and y_val is not None
        X_vt, y_vt, _ = _prep_tensors(X_val.copy(), y_val.copy(), scaler)
        val_loader = DataLoader(TensorDataset(X_vt, y_vt),
                                batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)

    # Loss, optimizer, scheduler
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=DL_LR_PATIENCE,
        factor=0.5, min_lr=DL_MIN_LR,
    )

    # Training
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(X_tr)
        history["train_loss"].append(train_loss)

        # Validation
        if has_val:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    out = model(xb)
                    val_loss += criterion(out, yb).item() * xb.size(0)
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_loss /= total
            val_acc = correct / total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info("  Epoch %3d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                            epoch, train_loss, val_loss, val_acc)

            if no_improve >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break
        else:
            history["val_loss"].append(train_loss)
            history["val_acc"].append(0.0)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("  Epoch %3d  train_loss=%.4f", epoch, train_loss)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "scaler": scaler, "best_val_loss": best_val_loss}


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   scaler, n_classes: int = 2) -> Dict:
    """Evaluate a trained PyTorch model → metrics dict."""
    model = model.to(DEVICE)
    model.eval()

    X_t, y_t, _ = _prep_tensors(X.copy(), y.copy(), scaler, fit=False)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_probs.append(prob)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred,
                                            average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred,
                                      average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred,
                              average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    # AUC
    try:
        if n_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob,
                                                      multi_class="ovr",
                                                      average="weighted"))
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Training approaches
# ══════════════════════════════════════════════════════════════════════════════

def _get_subject_data(data: Dict, window_sec: int, step_sec: int,
                      device_mode: str) -> Tuple[Dict, Optional[List[str]]]:
    """Extract features for each subject → {sid: (X, y)}."""
    sf: Dict = {}
    feat_names = None
    for subj in data["subjects"]:
        X, y, fn = extract_subject_features(subj, window_sec, step_sec,
                                             device_mode)
        if len(X):
            sf[subj["subject_id"]] = (X, y)
            feat_names = feat_names or fn
    return sf, feat_names


# ── LOSO ──────────────────────────────────────────────────────────────────────

def train_dl_loso(data: Dict, *, arch: str = "cnn1d",
                  n_classes: int = 2, window_sec: float = 60,
                  step_sec: float = 30, device_mode: str = "both",
                  **train_kw) -> Dict:
    """Leave-One-Subject-Out evaluation for one DL architecture."""
    sf, feat_names = _get_subject_data(data, window_sec, step_sec, device_mode)
    sids = sorted(sf)
    if len(sids) < 2:
        logger.error("LOSO requires at least 2 subjects.")
        return {}
    n_features = sf[sids[0]][0].shape[1]
    logger.info("LOSO %s  (%d subjects, %d features, %d classes)",
                arch, len(sids), n_features, n_classes)

    fold_results = []
    for test_sid in sids:
        X_te, y_te = sf[test_sid]
        X_tr = np.vstack([sf[s][0] for s in sids if s != test_sid])
        y_tr = np.concatenate([sf[s][1] for s in sids if s != test_sid])

        # Build a validation split only from training subjects (no test leakage).
        X_fit, y_fit, X_val, y_val = _split_train_val(X_tr, y_tr)
        cw = _compute_class_weights(y_fit)

        model = build_dl_model(arch, n_features, n_classes, DL_DROPOUT)
        info = train_model(model, X_fit, y_fit, X_val, y_val,
                           class_weights=cw, **train_kw)
        metrics = evaluate_model(model, X_te, y_te, info["scaler"], n_classes)
        metrics["test_subject"] = test_sid
        fold_results.append(metrics)
        logger.info("  LOSO S%d → Acc=%.4f  F1=%.4f  AUC=%.4f",
                    test_sid, metrics["accuracy"], metrics["f1"],
                    metrics.get("roc_auc", float("nan")))

    # Aggregate
    acc = [f["accuracy"] for f in fold_results]
    f1s = [f["f1"] for f in fold_results]
    summary = {
        "arch": arch,
        "n_classes": n_classes,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "per_subject": fold_results,
    }
    return summary


# ── Subject-Independent ──────────────────────────────────────────────────────

def train_dl_subject_independent(data: Dict, *, arch: str = "cnn1d",
                                  n_classes: int = 2,
                                  window_sec: float = 60, step_sec: float = 30,
                                  device_mode: str = "both",
                                  test_ratio: float = 0.2,
                                  **train_kw) -> Dict:
    """Pool subjects, split by subject (80/20), train & evaluate."""
    sf, feat_names = _get_subject_data(data, window_sec, step_sec, device_mode)
    sids = sorted(sf)
    if len(sids) < 2:
        logger.error("Subject-independent training requires at least 2 subjects.")
        return {}

    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(sids)
    n_test = max(1, int(len(sids) * test_ratio))
    test_set = sids[:n_test]
    train_set = sids[n_test:]

    X_tr = np.vstack([sf[s][0] for s in train_set])
    y_tr = np.concatenate([sf[s][1] for s in train_set])
    X_te = np.vstack([sf[s][0] for s in test_set])
    y_te = np.concatenate([sf[s][1] for s in test_set])

    X_fit, y_fit, X_val, y_val = _split_train_val(X_tr, y_tr)
    n_features = X_fit.shape[1]
    cw = _compute_class_weights(y_fit)
    logger.info("Subject-Independent %s  train=%s (%d) test=%s (%d)",
                arch, train_set, len(X_tr), test_set, len(X_te))

    model = build_dl_model(arch, n_features, n_classes, DL_DROPOUT)
    info = train_model(model, X_fit, y_fit, X_val, y_val,
                       class_weights=cw, **train_kw)
    metrics = evaluate_model(model, X_te, y_te, info["scaler"], n_classes)

    # Save model
    tag = f"{arch}_independent_{'bin' if n_classes == 2 else '3cls'}"
    save_dl_model(model, str(MODELS_DIR / f"{tag}.pt"),
                  meta={"train_subjects": train_set, "test_subjects": test_set},
                  scaler=info["scaler"])
    metrics.update({
        "arch": arch,
        "train_subjects": [int(s) for s in train_set],
        "test_subjects": [int(s) for s in test_set],
    })
    return metrics


# ── Subject-Dependent ─────────────────────────────────────────────────────────

def train_dl_subject_dependent(data: Dict, *, arch: str = "cnn1d",
                                n_classes: int = 2,
                                window_sec: float = 60, step_sec: float = 30,
                                device_mode: str = "both",
                                **train_kw) -> Dict:
    """Per-subject 80/20 split training & evaluation."""
    sf, _ = _get_subject_data(data, window_sec, step_sec, device_mode)
    results: Dict = {}

    for sid in sorted(sf):
        X, y = sf[sid]
        if len(X) < 10:
            logger.warning("  S%d: too few windows (%d), skipped", sid, len(X))
            continue
        n_features = X.shape[1]

        # 80/20 split
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.permutation(len(X))
        split = int(len(X) * 0.8)
        X_tr, y_tr = X[idx[:split]], y[idx[:split]]
        X_te, y_te = X[idx[split:]], y[idx[split:]]

        X_fit, y_fit, X_val, y_val = _split_train_val(X_tr, y_tr)
        cw = _compute_class_weights(y_fit)
        model = build_dl_model(arch, n_features, n_classes, DL_DROPOUT)
        info = train_model(model, X_fit, y_fit, X_val, y_val,
                           class_weights=cw, **train_kw)
        metrics = evaluate_model(model, X_te, y_te, info["scaler"], n_classes)
        results[f"S{sid}"] = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc", float("nan")),
            "n_windows": len(X),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Full comparison: ML + DL
# ══════════════════════════════════════════════════════════════════════════════

def compare_all(data: Dict, *, n_classes: int = 2,
                window_sec: float = 60, step_sec: float = 30,
                device_mode: str = "both") -> Dict:
    """Run LOSO for all ML + DL models and build comparison table."""
    from ml_models import StressModel  # local import to avoid circular

    sf, feat_names = _get_subject_data(data, window_sec, step_sec, device_mode)
    sids = sorted(sf)
    if len(sids) < 2:
        logger.error("Comparison requires at least 2 subjects.")
        return {}

    comparison: Dict = {"ml": {}, "dl": {}}

    # ── ML baselines ──────────────────────────────────────────────────────
    for name, params in DEFAULT_MODELS.items():
        fold_acc, fold_f1 = [], []
        for test_sid in sids:
            X_te, y_te = sf[test_sid]
            X_tr = np.vstack([sf[s][0] for s in sids if s != test_sid])
            y_tr = np.concatenate([sf[s][1] for s in sids if s != test_sid])
            mdl = StressModel(name, params)
            try:
                mdl.fit(X_tr, y_tr, verbose=False)
                m = mdl.evaluate(X_te, y_te, verbose=False)
                fold_acc.append(m["accuracy"])
                fold_f1.append(m["f1"])
            except Exception as e:
                logger.error("ML %s S%d: %s", name, test_sid, e)
        if fold_acc:
            comparison["ml"][name] = {
                "accuracy_mean": float(np.mean(fold_acc)),
                "accuracy_std": float(np.std(fold_acc)),
                "f1_mean": float(np.mean(fold_f1)),
                "f1_std": float(np.std(fold_f1)),
            }

    # ── DL models ─────────────────────────────────────────────────────────
    for arch in DL_MODELS:
        result = train_dl_loso(data, arch=arch, n_classes=n_classes,
                               window_sec=window_sec, step_sec=step_sec,
                               device_mode=device_mode)
        if result:
            comparison["dl"][arch] = {
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "f1_mean": result["f1_mean"],
                "f1_std": result["f1_std"],
            }

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(results: Dict, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RESULTS_DIR / f"dl_{tag}_{ts}.json"
    p.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(p, "w") as f:
        json.dump(results, f, indent=2, default=_default)
    logger.info("Results → %s", p)
    return p


def _print_comparison(comp: Dict):
    print(f"\n{'='*70}")
    print(f"  {'Model':20s} {'Accuracy':>18s} {'F1':>18s}")
    print(f"{'='*70}")
    for section, models in comp.items():
        print(f"\n  [{section.upper()}]")
        for name, m in models.items():
            acc = f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}"
            f1 = f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
            print(f"  {name:20s} {acc:>18s} {f1:>18s}")


def _print_loso(result: Dict):
    print(f"\n{'='*65}")
    print(f"  LOSO  {result.get('arch', '?')}  "
          f"({result.get('n_classes', '?')}-class)")
    print(f"{'='*65}")
    print(f"  Accuracy: {result['accuracy_mean']:.4f} "
          f"± {result['accuracy_std']:.4f}")
    print(f"  F1:       {result['f1_mean']:.4f} "
          f"± {result['f1_std']:.4f}")
    for f in result.get("per_subject", []):
        print(f"    S{f['test_subject']:2d}  "
              f"Acc={f['accuracy']:.4f}  F1={f['f1']:.4f}  "
              f"AUC={f.get('roc_auc', float('nan')):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="DL stress detection training")
    ap.add_argument("--arch", default="cnn1d",
                    choices=DL_MODELS + ["all"],
                    help="DL architecture (or 'all')")
    ap.add_argument("--approach", default="loso",
                    choices=["subject_dependent", "subject_independent",
                             "loso", "compare", "all"])
    ap.add_argument("--classes", default="binary",
                    choices=["binary", "3class", "both"],
                    help=("binary = stress vs non-stress (baseline+amusement), "
                          "3class = baseline/stress/amusement"))
    ap.add_argument("--device", default="both",
                    choices=["wrist", "chest", "both"],
                    help="Sensor device(s) to use")
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--window", type=float, default=WINDOW_SIZE)
    ap.add_argument("--step", type=float, default=WINDOW_STEP)
    ap.add_argument("--paper-protocol", action="store_true",
                    help="Use WESAD paper setup: 60s window, 0.25s shift")
    ap.add_argument("--epochs", type=int, default=DL_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DL_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DL_LEARNING_RATE)
    ap.add_argument("--wesad-dir", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("Device: %s  |  CUDA available: %s", DEVICE,
                torch.cuda.is_available())

    class_configs = []
    if args.classes in ("binary", "both"):
        class_configs.append(2)
    if args.classes in ("3class", "both"):
        class_configs.append(3)

    archs = DL_MODELS if args.arch == "all" else [args.arch]
    train_kw = dict(epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr)

    if args.paper_protocol:
        window_sec, step_sec = 60.0, 0.25
        logger.info("Using paper protocol: window=%.2fs, step=%.2fs",
                    window_sec, step_sec)
    else:
        window_sec, step_sec = args.window, args.step

    for n_cls in class_configs:
        logger.info("\n%s  %d-class  %s", "#" * 20, n_cls, "#" * 20)
        data = load_wesad(subject_ids=args.subjects,
                          wesad_dir=args.wesad_dir,
                          device=args.device, n_classes=n_cls)
        if not data["subjects"]:
            logger.error("No subjects loaded.")
            continue

        approaches = (["subject_dependent", "subject_independent", "loso"]
                      if args.approach == "all"
                      else ["compare"] if args.approach == "compare"
                      else [args.approach])

        for approach in approaches:
            if approach == "compare":
                comp = compare_all(data, n_classes=n_cls,
                                   window_sec=window_sec,
                                   step_sec=step_sec,
                                   device_mode=args.device)
                _print_comparison(comp)
                _save(comp, f"compare_{n_cls}cls")

            elif approach == "loso":
                for arch in archs:
                    r = train_dl_loso(data, arch=arch, n_classes=n_cls,
                                     window_sec=window_sec,
                                     step_sec=step_sec,
                                     device_mode=args.device,
                                     **train_kw)
                    if r:
                        _print_loso(r)
                        _save(r, f"loso_{arch}_{n_cls}cls")

            elif approach == "subject_independent":
                for arch in archs:
                    r = train_dl_subject_independent(
                        data, arch=arch, n_classes=n_cls,
                        window_sec=window_sec, step_sec=step_sec,
                        device_mode=args.device, **train_kw)
                    if r:
                        _save(r, f"independent_{arch}_{n_cls}cls")

            elif approach == "subject_dependent":
                for arch in archs:
                    r = train_dl_subject_dependent(
                        data, arch=arch, n_classes=n_cls,
                        window_sec=window_sec, step_sec=step_sec,
                        device_mode=args.device, **train_kw)
                    if r:
                        _save(r, f"dependent_{arch}_{n_cls}cls")

    logger.info("DL training complete.")


if __name__ == "__main__":
    main()
