"""
Standalone evaluation script — called by the Airflow evaluate_model task.

Loads the best.pth artifact from an MLflow run, runs inference on the
test split, and writes a report JSON + logs metrics back to the run.

Usage:
    python -m src.pipeline.evaluate_mlflow \
        --tracking-uri http://mlflow:5000 \
        --run-id abc123 \
        --split-dir data/split/v0003 \
        --seq-len 30 \
        --output /tmp/eval_report.json
"""

import os
import json
import argparse
import tempfile
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.model.data_loader import SignLanguageDataset
from src.model.train import build_model, load_checkpoint

logger = logging.getLogger(__name__)


def evaluate(
    tracking_uri: str,
    run_id: str,
    split_dir: str,
    seq_len: int,
    output_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    run    = client.get_run(run_id)
    params = run.data.params

    # Reconstruct model architecture from logged params
    model_type  = params.get("model_type",  "bilstm")
    input_dim   = int(params.get("input_dim",   225))
    hidden_dim  = int(params.get("hidden_dim",  256))
    num_layers  = int(params.get("num_layers",  2))
    dropout     = float(params.get("dropout",   0.3))
    label_map   = json.loads(params.get("label_map", "[]"))
    num_classes = len(label_map)

    if num_classes == 0:
        raise ValueError(f"label_map is empty in run {run_id}")

    # Download best.pth artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        local_ckpt = client.download_artifacts(run_id, "checkpoints/best.pth", tmpdir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = build_model(model_type, input_dim, hidden_dim,
                             num_classes, num_layers, dropout).to(device)
        load_checkpoint(local_ckpt, model, device=device)

    model.eval()

    test_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(test_dir):
        logger.warning(f"No test split at {test_dir} — skipping evaluation.")
        return {"test_acc": None, "report": None}

    test_ds = SignLanguageDataset(
        split_dir,
        seq_len=seq_len,
        source="npy",
        split="test",
        normalize=True,
        augment=False,
        label_map=label_map,
    )

    if len(test_ds) == 0:
        logger.warning("Test dataset is empty.")
        return {"test_acc": None, "report": None}

    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())

    test_acc = accuracy_score(labels, preds)
    report   = classification_report(
        labels, preds, target_names=label_map, output_dict=True
    )
    report_text = classification_report(labels, preds, target_names=label_map)

    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"\n{report_text}")

    # Log back to the MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_acc", test_acc)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as tmp:
            tmp.write(report_text)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="evaluation")
        os.unlink(tmp_path)

    result = {"test_acc": test_acc, "report": report, "run_id": run_id}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", required=True)
    p.add_argument("--run-id",       required=True)
    p.add_argument("--split-dir",    required=True)
    p.add_argument("--seq-len",      type=int, default=30)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--num-workers",  type=int, default=0)
    p.add_argument("--output",       default="/tmp/eval_report.json")
    args = p.parse_args()

    result = evaluate(
        tracking_uri=args.tracking_uri,
        run_id=args.run_id,
        split_dir=args.split_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_path=args.output,
    )
    print(f"test_acc={result.get('test_acc')}")
