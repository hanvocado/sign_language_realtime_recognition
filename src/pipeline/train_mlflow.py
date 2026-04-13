"""
MLflow-integrated training script for Sign Language Recognition.

Wraps train.py logic with:
  - mlflow.start_run() context
  - Auto-logging of params, metrics per epoch, and artifacts
  - Fine-tune support: loads weights from a base checkpoint URI
  - label_map logged as a JSON param (for retrain_check.py to read)
  - Saves best.pth as an MLflow artifact

Usage examples
──────────────
# Full train from scratch
python -m src.pipeline.train_mlflow \
    --data-dir data/split \
    --model-name SLR_model \
    --tracking-uri http://localhost:5000 \
    --experiment-name slr_v1

# Fine-tune from Production checkpoint
python -m src.pipeline.train_mlflow \
    --data-dir data/split \
    --model-name SLR_model \
    --tracking-uri http://localhost:5000 \
    --experiment-name slr_v1 \
    --base-ckpt-uri mlflow-artifacts:/1/abc123/artifacts/checkpoints/best.pth \
    --finetune-lr 0.0005
"""

import os
import json
import argparse
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Re-use existing project modules
from src.model.data_loader import SignLanguageDataset
from src.model.train import (
    build_model, EarlyStopping, save_checkpoint, load_checkpoint,
    str2bool,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Checkpoint download helper
# ─────────────────────────────────────────────

def download_base_checkpoint(ckpt_uri: str, local_path: str = "/tmp/base_best.pth"):
    """
    Download an artifact from MLflow artifact store to local disk.

    Supports MLflow artifact URIs such as:
      - mlflow-artifacts:/<exp_id>/<run_id>/artifacts/<path>
      - mlflow-artifacts://<exp_id>/<run_id>/artifacts/<path>
      - runs:/<run_id>/<path>
    """
    client = MlflowClient()

    run_id = None
    artifact_path = None

    if ckpt_uri.startswith("runs:/"):
        uri = ckpt_uri[len("runs:/"):].lstrip("/")
        parts = [part for part in uri.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Cannot parse runs URI: {ckpt_uri}")
        run_id = parts[0]
        artifact_path = "/".join(parts[1:])
    elif ckpt_uri.startswith("mlflow-artifacts://") or ckpt_uri.startswith("mlflow-artifacts:/"):
        if ckpt_uri.startswith("mlflow-artifacts://"):
            uri = ckpt_uri[len("mlflow-artifacts://"):]
        else:
            uri = ckpt_uri[len("mlflow-artifacts:/"):]
        parts = [part for part in uri.lstrip("/").split("/") if part]
        # Expected format: <exp_id>/<run_id>/artifacts/<path>
        if len(parts) < 4 or parts[2] != "artifacts":
            raise ValueError(f"Cannot parse artifact URI: {ckpt_uri}")
        run_id = parts[1]
        artifact_path = "/".join(parts[3:])
    else:
        raise ValueError(f"Unsupported checkpoint URI scheme: {ckpt_uri}")

    if not run_id or not artifact_path:
        raise ValueError(
            f"Invalid checkpoint URI: expected non-empty run_id and artifact_path, got "
            f"run_id={run_id!r}, artifact_path={artifact_path!r} from {ckpt_uri}"
        )

    dst_dir = os.path.dirname(local_path)
    os.makedirs(dst_dir, exist_ok=True)
    downloaded = client.download_artifacts(run_id, artifact_path, dst_dir)
    return downloaded


# ─────────────────────────────────────────────
# Training loop (with per-epoch MLflow logging)
# ─────────────────────────────────────────────

def run_training(args, model, train_loader, val_loader, label_map,
                 device, num_classes, run_id: str):
    """
    Training loop that logs metrics to the active MLflow run each epoch.
    Returns best_val_acc and history dict.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    lr = args.finetune_lr if args.finetune_lr else args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    early_stopping = EarlyStopping(patience=args.patience)

    ckpt_dir = os.path.join(args.ckpt_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = 0.0
    best_epoch   = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        t_losses, t_preds, t_labels = [], [], []
        t0 = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            if torch.isnan(loss):
                logger.error(f"NaN loss at epoch {epoch} — aborting.")
                return best_val_acc, history

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_losses.append(loss.item())
            t_preds.extend(logits.argmax(1).cpu().tolist())
            t_labels.extend(y.cpu().tolist())

        avg_tl = float(np.mean(t_losses))
        t_acc  = accuracy_score(t_labels, t_preds)

        # ── Val ──
        model.eval()
        v_losses, v_preds, v_labels = [], [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                v_losses.append(criterion(logits, y).item())
                v_preds.extend(logits.argmax(1).cpu().tolist())
                v_labels.extend(y.cpu().tolist())

        avg_vl = float(np.mean(v_losses))
        v_acc  = accuracy_score(v_labels, v_preds)
        elapsed = time.time() - t0

        scheduler.step(v_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── MLflow per-epoch metrics ──
        mlflow.log_metrics({
            "train_loss": avg_tl,
            "train_acc":  t_acc,
            "val_loss":   avg_vl,
            "val_acc":    v_acc,
            "lr":         current_lr,
        }, step=epoch)

        history["train_loss"].append(avg_tl)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(avg_vl)
        history["val_acc"].append(v_acc)

        logger.info(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"tLoss {avg_tl:.4f} | tAcc {t_acc:.4f} | "
            f"vLoss {avg_vl:.4f} | vAcc {v_acc:.4f} | "
            f"lr {current_lr:.6f} | {elapsed:.1f}s"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            ckpt_path = os.path.join(ckpt_dir, "best.pth")
            save_checkpoint(
                model, optimizer, epoch, ckpt_path,
                val_acc=v_acc,
                label_map=label_map,
                args=vars(args),
            )
            logger.info(f"  ✓ Best checkpoint saved (epoch {epoch}, val_acc {v_acc:.4f})")

        if early_stopping(v_acc):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    mlflow.log_metric("best_val_acc",  best_val_acc)
    mlflow.log_metric("best_epoch",    best_epoch)

    return best_val_acc, history


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # ── Datasets ──
    augment_config = {
        "rotation_range": args.rotation_range,
        "scale_range":    (args.scale_min, args.scale_max),
        "shift_range":    args.shift_range,
        "flip_prob":      args.flip_prob,
        "time_mask_prob": args.time_mask_prob,
        "time_mask_max":  args.time_mask_max,
    }

    train_ds = SignLanguageDataset(
        args.data_dir, seq_len=args.seq_len, source="npy", split="train",
        normalize=args.normalize, augment=args.augment_train,
        augment_config=augment_config,
    )
    label_map    = train_ds.get_label_map()
    num_classes  = len(label_map)

    val_ds = SignLanguageDataset(
        args.data_dir, seq_len=args.seq_len, source="npy", split="val",
        normalize=args.normalize, augment=False, label_map=label_map,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # ── Model ──
    model = build_model(
        args.model_type, args.input_dim, args.hidden_dim,
        num_classes, args.num_layers, args.dropout,
    ).to(device)

    run_tags = {
        "mode":        "finetune" if args.base_ckpt_uri else "full",
        "model_type":  args.model_type,
        "num_classes": str(num_classes),
    }

    if args.base_ckpt_uri:
        logger.info(f"Fine-tuning from: {args.base_ckpt_uri}")
        local_ckpt = download_base_checkpoint(args.base_ckpt_uri)
        checkpoint = torch.load(local_ckpt, map_location=device)
        # Partial load: skip classifier head if num_classes changed
        state = checkpoint["model_state"]
        curr_state = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in curr_state and v.shape == curr_state[k].shape}
        skipped = set(state.keys()) - set(compatible.keys())
        if skipped:
            logger.info(f"Skipped layers (shape mismatch / new head): {skipped}")
        curr_state.update(compatible)
        model.load_state_dict(curr_state)
        run_tags["base_run_id"] = checkpoint.get("run_id", "unknown")

    # ── MLflow run ──
    run_tags["model_name"] = args.model_name
    with mlflow.start_run(run_name=args.run_name, tags=run_tags) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log all hyperparams
        mlflow.log_params({
            "model_type":      args.model_type,
            "input_dim":       args.input_dim,
            "hidden_dim":      args.hidden_dim,
            "num_layers":      args.num_layers,
            "dropout":         args.dropout,
            "seq_len":         args.seq_len,
            "batch_size":      args.batch_size,
            "lr":              args.finetune_lr or args.lr,
            "weight_decay":    args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "epochs":          args.epochs,
            "patience":        args.patience,
            "normalize":       args.normalize,
            "augment_train":   args.augment_train,
            # Dataset info — retrain_check.py reads these
            "num_classes":     num_classes,
            "train_samples":   len(train_ds),
            "val_samples":     len(val_ds),
            "label_map":       json.dumps(label_map),
        })

        best_val_acc, history = run_training(
            args, model, train_loader, val_loader,
            label_map, device, num_classes, run_id,
        )

        # ── Log artifacts ──
        ckpt_dir = os.path.join(args.ckpt_dir, run_id)
        ckpt_path = os.path.join(ckpt_dir, "best.pth")

        if os.path.exists(ckpt_path):
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            # Load best weights back into model before logging to MLflow registry
            best_ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state"])

        # Save and log label map
        lm_path = os.path.join(ckpt_dir, "label_map.json")
        with open(lm_path, "w") as f:
            json.dump(label_map, f)
        mlflow.log_artifact(lm_path, artifact_path="checkpoints")

        # Save and log training history
        hist_path = os.path.join(ckpt_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)
        mlflow.log_artifact(hist_path, artifact_path="checkpoints")

        # Log the PyTorch model for registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=None,  # promote separately
        )

        logger.info(f"Training complete. best_val_acc={best_val_acc:.4f}")

        # Write run_id to output file if requested (for deterministic retrieval by caller)
        if args.run_id_output_file:
            out_dir = os.path.dirname(args.run_id_output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.run_id_output_file, "w") as f:
                f.write(run_id)

    return run_id, best_val_acc


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MLflow-integrated SLR training")

    # MLflow
    p.add_argument("--tracking-uri",    required=True)
    p.add_argument("--experiment-name", default="slr_experiment")
    p.add_argument("--model-name",      default="SLR_model",
                   help="Registered model name (used by promote_model.py)")
    p.add_argument("--run-name", default=None,
                   help="Optional MLflow run display name")
    p.add_argument("--run-id-output-file", default=None,
                   help="Path to write the MLflow run_id on completion (for caller retrieval)")

    # Fine-tune
    p.add_argument("--base-ckpt-uri", default=None,
                   help="mlflow-artifacts URI of base checkpoint for fine-tuning")
    p.add_argument("--finetune-lr",   type=float, default=None,
                   help="Override learning rate for fine-tuning (default: use --lr)")

    # Data
    p.add_argument("--data-dir",    default="data/split")
    p.add_argument("--seq-len",     type=int, default=30)
    p.add_argument("--input-dim",   type=int, default=225)
    p.add_argument("--num-workers", type=int, default=0)

    # Model
    p.add_argument("--model-type",  choices=["lstm", "bilstm", "gru"], default="bilstm")
    p.add_argument("--hidden-dim",  type=int, default=256)
    p.add_argument("--num-layers",  type=int, default=2)
    p.add_argument("--dropout",     type=float, default=0.3)

    # Training
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--lr",               type=float, default=0.001)
    p.add_argument("--weight-decay",     type=float, default=1e-3)
    p.add_argument("--label-smoothing",  type=float, default=0.1)
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--ckpt-dir",         default="models/checkpoints")

    # Augmentation
    p.add_argument("--normalize",       type=str2bool, default=True)
    p.add_argument("--augment-train",   type=str2bool, default=True)
    p.add_argument("--rotation-range",  type=float, default=15.0)
    p.add_argument("--scale-min",       type=float, default=0.85)
    p.add_argument("--scale-max",       type=float, default=1.15)
    p.add_argument("--shift-range",     type=float, default=0.08)
    p.add_argument("--flip-prob",       type=float, default=0.5)
    p.add_argument("--time-mask-prob",  type=float, default=0.2)
    p.add_argument("--time-mask-max",   type=int,   default=3)

    args = p.parse_args()

    # Normalise dashes→underscores for downstream code that uses vars(args)
    args.seq_len        = args.seq_len
    args.input_dim      = args.input_dim
    args.hidden_dim     = args.hidden_dim
    args.num_layers     = args.num_layers
    args.model_type     = args.model_type
    args.batch_size     = args.batch_size
    args.weight_decay   = args.weight_decay
    args.label_smoothing= args.label_smoothing
    args.num_workers    = args.num_workers
    args.ckpt_dir       = args.ckpt_dir
    args.augment_train  = args.augment_train
    args.rotation_range = args.rotation_range
    args.scale_min      = args.scale_min
    args.scale_max      = args.scale_max
    args.shift_range    = args.shift_range
    args.flip_prob      = args.flip_prob
    args.time_mask_prob = args.time_mask_prob
    args.time_mask_max  = args.time_mask_max
    args.base_ckpt_uri  = args.base_ckpt_uri
    args.finetune_lr    = args.finetune_lr

    run_id, best_val_acc = main(args)
    print(f"\nrun_id={run_id}  best_val_acc={best_val_acc:.4f}")
