"""Raw-signal DL baseline (LOSO) for WESAD stress detection.

This script trains a simple multi-channel 1D-CNN directly on windowed
physiological waveforms (after signal-level preprocessing and resampling),
providing a true raw-signal baseline complementary to feature-based DL.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DL_BATCH_SIZE,
    DL_DROPOUT,
    DL_EPOCHS,
    DL_LEARNING_RATE,
    DL_LR_PATIENCE,
    DL_MIN_LR,
    DL_PATIENCE,
    DL_WEIGHT_DECAY,
    RANDOM_STATE,
    RESULTS_DIR,
    WINDOW_SIZE,
    WINDOW_STEP,
)
from raw_signal import extract_subject_raw_windows
from wesad_loader import load_wesad

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RawStressCNN1D(nn.Module):
    """Compact multi-channel 1D CNN for raw waveform classification."""

    def __init__(self, n_channels: int, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split train/val robustly for small or imbalanced sets."""
    if len(X) < 10 or val_ratio <= 0:
        return X, y, None, None

    uniq, cnt = np.unique(y, return_counts=True)
    stratify = y if len(uniq) > 1 and cnt.min() >= 2 else None
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return X, y, None, None

    if len(np.unique(y_tr)) < 2 or len(X_val) == 0:
        return X, y, None, None
    return X_tr, y_tr, X_val, y_val


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts.astype(float))
    out = torch.zeros(int(classes.max()) + 1, dtype=torch.float32)
    for c, w in zip(classes, weights):
        out[int(c)] = float(w)
    return out


def _fit_channel_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-channel normalization using train-set statistics."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_channel_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_raw_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    class_weights: Optional[torch.Tensor],
) -> Dict:
    """Train raw-signal model with optional validation and early stopping."""
    model = model.to(DEVICE)

    norm_mean, norm_std = _fit_channel_norm(X_train)
    X_tr = _apply_channel_norm(X_train, norm_mean, norm_std)
    train_loader = _to_loader(X_tr, y_train, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    if has_val:
        X_v = _apply_channel_norm(X_val, norm_mean, norm_std)
        val_loader = _to_loader(X_v, y_val, batch_size=batch_size, shuffle=False)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=DL_LR_PATIENCE,
        factor=0.5,
        min_lr=DL_MIN_LR,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = running / max(n_seen, 1)
        history["train_loss"].append(float(train_loss))

        if has_val:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * xb.size(0)
                    pred = logits.argmax(dim=1)
                    correct += int((pred == yb).sum().item())
                    total += int(yb.size(0))

            val_loss /= max(total, 1)
            val_acc = correct / max(total, 1)
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %3d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_acc,
                )

            if no_improve >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break
        else:
            history["val_loss"].append(float(train_loss))
            history["val_acc"].append(0.0)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("  Epoch %3d  train_loss=%.4f", epoch, train_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


def evaluate_raw_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> Dict:
    """Evaluate raw DL model and return standard metrics dict."""
    model = model.to(DEVICE)
    model.eval()

    Xn = _apply_channel_norm(X, norm_mean, norm_std)
    loader = _to_loader(Xn, y, batch_size=256, shuffle=False)

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            all_true.append(yb.numpy())
            all_pred.append(pred)
            all_prob.append(prob)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    try:
        if n_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def train_raw_loso(
    data: Dict,
    *,
    n_classes: int,
    device_mode: str,
    window_sec: float,
    step_sec: float,
    target_sr: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Dict:
    """Run LOSO with raw-signal CNN baseline."""
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    channel_names: Optional[List[str]] = None

    for subj in data["subjects"]:
        X, y, channels = extract_subject_raw_windows(
            subj,
            device_mode=device_mode,
            window_sec=window_sec,
            step_sec=step_sec,
            target_sr=target_sr,
        )
        if len(X) == 0:
            continue
        sid = int(subj["subject_id"])
        subject_data[sid] = (X, y)
        channel_names = channel_names or channels
        logger.info("  S%d: %d windows, channels=%d, timesteps=%d", sid, len(X), X.shape[1], X.shape[2])

    sids = sorted(subject_data)
    if len(sids) < 2:
        logger.error("Raw LOSO requires at least 2 subjects with valid windows.")
        return {}

    n_channels = int(subject_data[sids[0]][0].shape[1])
    n_timesteps = int(subject_data[sids[0]][0].shape[2])
    logger.info(
        "Raw LOSO rawcnn1d (%d subjects, %d channels, %d timesteps, %d classes)",
        len(sids),
        n_channels,
        n_timesteps,
        n_classes,
    )

    fold_results: List[Dict] = []
    for test_sid in sids:
        X_te, y_te = subject_data[test_sid]
        X_tr = np.concatenate([subject_data[s][0] for s in sids if s != test_sid], axis=0)
        y_tr = np.concatenate([subject_data[s][1] for s in sids if s != test_sid], axis=0)

        X_fit, y_fit, X_val, y_val = _split_train_val(X_tr, y_tr)
        class_weights = _compute_class_weights(y_fit)

        model = RawStressCNN1D(n_channels=n_channels, n_classes=n_classes, dropout=DL_DROPOUT)
        info = train_raw_model(
            model,
            X_fit,
            y_fit,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            class_weights=class_weights,
        )
        metrics = evaluate_raw_model(
            model,
            X_te,
            y_te,
            n_classes=n_classes,
            norm_mean=info["norm_mean"],
            norm_std=info["norm_std"],
        )
        metrics["test_subject"] = int(test_sid)
        fold_results.append(metrics)

        logger.info(
            "  LOSO S%d -> Acc=%.4f  F1=%.4f  AUC=%.4f",
            test_sid,
            metrics["accuracy"],
            metrics["f1"],
            metrics.get("roc_auc", float("nan")),
        )

    acc = [f["accuracy"] for f in fold_results]
    f1 = [f["f1"] for f in fold_results]

    return {
        "arch": "rawcnn1d",
        "input_mode": "raw_signal",
        "n_classes": int(n_classes),
        "device": device_mode,
        "target_sr": int(target_sr),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "n_channels": n_channels,
        "n_timesteps": n_timesteps,
        "channels": channel_names or [],
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "f1_mean": float(np.mean(f1)),
        "f1_std": float(np.std(f1)),
        "per_subject": fold_results,
    }


def _save(results: Dict, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"dl_{tag}_{ts}.json"

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_default)
    logger.info("Results -> %s", out)
    return out


def _print_loso(result: Dict) -> None:
    print("\n" + "=" * 65)
    print(f"  RAW LOSO rawcnn1d ({result.get('n_classes', '?')}-class, {result.get('device', '?')})")
    print("=" * 65)
    print(f"  Accuracy: {result['accuracy_mean']:.4f} +/- {result['accuracy_std']:.4f}")
    print(f"  F1:       {result['f1_mean']:.4f} +/- {result['f1_std']:.4f}")
    for fold in result.get("per_subject", []):
        print(
            f"    S{int(fold['test_subject']):2d}  "
            f"Acc={fold['accuracy']:.4f}  F1={fold['f1']:.4f}  "
            f"AUC={fold.get('roc_auc', float('nan')):.4f}"
        )


def _with_meta(result: Dict, meta: Dict) -> Dict:
    out = dict(result)
    out["_meta"] = dict(meta)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw-signal DL baseline (LOSO)")
    ap.add_argument("--classes", default="binary", choices=["binary", "3class", "both"])
    ap.add_argument("--device", default="both", choices=["wrist", "chest", "both"])
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--window", type=float, default=WINDOW_SIZE)
    ap.add_argument("--step", type=float, default=WINDOW_STEP)
    ap.add_argument("--paper-protocol", action="store_true")
    ap.add_argument("--target-sr", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=DL_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DL_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DL_LEARNING_RATE)
    ap.add_argument("--wesad-dir", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("Device: %s | CUDA available: %s", DEVICE, torch.cuda.is_available())

    if args.paper_protocol:
        window_sec, step_sec = 60.0, 0.25
        logger.info("Using paper protocol: window=%.2fs, step=%.2fs", window_sec, step_sec)
    else:
        window_sec, step_sec = args.window, args.step

    class_configs: List[int] = []
    if args.classes in ("binary", "both"):
        class_configs.append(2)
    if args.classes in ("3class", "both"):
        class_configs.append(3)

    for n_classes in class_configs:
        logger.info("\n%s  %d-class  %s", "#" * 20, n_classes, "#" * 20)
        data = load_wesad(
            subject_ids=args.subjects,
            wesad_dir=args.wesad_dir,
            device=args.device,
            n_classes=n_classes,
        )
        if not data.get("subjects"):
            logger.error("No subjects loaded.")
            continue

        res = train_raw_loso(
            data,
            n_classes=n_classes,
            device_mode=args.device,
            window_sec=window_sec,
            step_sec=step_sec,
            target_sr=args.target_sr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=DL_WEIGHT_DECAY,
            patience=DL_PATIENCE,
        )
        if not res:
            continue

        _print_loso(res)
        meta = {
            "pipeline": "dl_raw_signal",
            "approach": "loso",
            "device": args.device,
            "n_classes": int(n_classes),
            "window_sec": float(window_sec),
            "step_sec": float(step_sec),
            "paper_protocol": bool(args.paper_protocol),
            "target_sr": int(args.target_sr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "arch": "rawcnn1d",
        }
        tag = f"loso_rawcnn1d_{args.device}_{n_classes}cls"
        _save(_with_meta(res, meta), tag)

    logger.info("Raw-signal DL baseline complete.")


if __name__ == "__main__":
    main()
