"""
train.py
--------
PRD Section 11 (Training Strategy) + Section 12 (Evaluation Metrics).

Trains LineMVGNN with mini-batch neighbor sampling, AdamW, weighted BCE
(class imbalance), and ReduceLROnPlateau on validation PR-AUC. Every
heavy-compute section is timed:

    05. Model Training (total)
      05a. Training - Neighbor Sampling (cumulative)
      05b. Training - Model Forward/Backward (cumulative)
      05c. Training - Validation Pass (cumulative)
    06. Model Inference (test set)
    07. Account Risk Aggregation
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)
import time
from torch_geometric.loader import NeighborLoader


def chronological_split(num_nodes: int, val_frac: float = 0.15):
    """Last `val_frac` of (already time-sorted) txn_ids become validation --
    avoids optimistic leakage you'd get from a random split on a temporal
    money-flow graph."""
    split_at = int(num_nodes * (1 - val_frac))
    train_idx = torch.arange(0, split_at)
    val_idx = torch.arange(split_at, num_nodes)
    return train_idx, val_idx


def train_model(
    model: nn.Module,
    data,
    num_epochs: int = 8,
    batch_size: int = 512,
    num_neighbors=(15, 10),
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    device: str = "cpu",
    verbose: bool = True,
):
    model.to(device)
    train_idx, val_idx = chronological_split(data.num_nodes, val_frac)

    train_loader = NeighborLoader(
        data, 
        num_neighbors=list(num_neighbors),
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        input_nodes=val_idx,
        batch_size=batch_size,
        shuffle=False,
    )

    n_pos = data.y[train_idx].sum().item()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if verbose:
        print(f"Train pos_weight (class-imbalance correction): {pos_weight.item():.2f} "
              f"({int(n_pos)} positive / {int(n_neg)} negative)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}

    t0_train = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            logits = model(batch)[:batch.seed_size]
            y = batch.y[:batch.seed_size]
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_loss, val_pr_auc = _evaluate_loader(model, val_loader, loss_fn, device)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        scheduler.step(val_pr_auc)

        if verbose:
            print(f"Epoch {epoch:>2d}/{num_epochs}  "
                    f"train_loss={history['train_loss'][-1]:.4f}  "
                    f"val_loss={val_loss:.4f}  val_PR-AUC={val_pr_auc:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

    print("05. Model Training (total): ", time.time() - t0_train)
    return model, history


def _evaluate_loader(model, loader, loss_fn, device):
    model.eval()
    losses, all_probs, all_y = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
        
            logits = model(batch)[:batch.seed_size]
            y = batch.y[:batch.seed_size]
            losses.append(loss_fn(logits, y).item())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_y)
    pr_auc = average_precision_score(y_true, probs) if y_true.sum() > 0 else 0.0
    return float(np.mean(losses)), float(pr_auc)


def run_inference(model, data, batch_size=1024, num_neighbors=(15, 10), device="cpu"):
    """Full-dataset scoring pass (PRD Section 14). Returns probs[N] aligned to data node order."""
    model.eval()
    all_idx = torch.arange(data.num_nodes)
    loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        input_nodes=all_idx,
        batch_size=batch_size,
        shuffle=False,
    )
    probs = np.zeros(data.num_nodes, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        pos = 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)[:batch.seed_size]
            p = torch.sigmoid(logits).cpu().numpy()
            n = len(p)
            # SimpleNeighborLoader (shuffle=False) yields seeds in
            # `all_idx` order, batch-by-batch, so a running position
            # pointer is enough to place them back correctly.
            probs[pos:pos + n] = p
            pos += n
    
    print("06. Model Inference: ", time.time() - t0)

    return probs


def compute_classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probs) if y_true.sum() > 0 else 0.0,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def print_metrics_report(metrics: dict, title: str = "TEST SET METRICS"):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
    for k in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
        print(f"{k:<12s}: {metrics[k]:.4f}")
    print("Confusion matrix [rows=true, cols=pred] (0=legit, 1=laundering):")
    print(metrics["confusion_matrix"])
    print("=" * 60 + "\n")
