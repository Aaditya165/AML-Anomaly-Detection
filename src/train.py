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
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)
from .neighbor_sampling import SimpleNeighborLoader


def chronological_split(num_nodes: int, val_frac: float = 0.15):
    """Last `val_frac` of (already time-sorted) txn_ids become validation --
    avoids optimistic leakage you'd get from a random split on a temporal
    money-flow graph."""
    split_at = int(num_nodes * (1 - val_frac))
    train_idx = torch.arange(0, split_at)
    val_idx = torch.arange(split_at, num_nodes)
    return train_idx, val_idx


def subsample_negatives(y: torch.Tensor, idx: torch.Tensor, neg_per_pos: float = 10.0, seed: int = 0) -> torch.Tensor:
    """
    Keep ALL positive (laundering) seeds in `idx`, plus a random sample of
    negatives at `neg_per_pos` per positive. This only changes which nodes
    are used as TRAINING SEEDS (i.e. which nodes' loss gets backpropagated)
    -- it does NOT touch the graph itself, so a dropped negative
    transaction is still visible to the GNN as neighbor *context* for
    whichever seeds it's a predecessor of. Standard technique for heavily
    imbalanced graphs; at ~0.1% laundering rate this can cut batches/epoch
    by two orders of magnitude with little to no loss in model quality,
    since the dropped negatives were almost all redundant easy-negatives.
    """
    rng = np.random.default_rng(seed)
    y_idx = y[idx]
    pos_idx = idx[y_idx == 1]
    neg_idx = idx[y_idx == 0]
    n_keep_neg = min(len(neg_idx), int(len(pos_idx) * neg_per_pos))
    keep_neg = neg_idx[torch.from_numpy(rng.choice(len(neg_idx), size=n_keep_neg, replace=False))]
    out = torch.cat([pos_idx, keep_neg])
    return out[torch.randperm(len(out))]


def train_model(
    model: nn.Module,
    data,
    predecessors,
    num_epochs: int = 8,
    batch_size: int = 512,
    num_neighbors=(25, 15),
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    neg_per_pos: float = None,
    device: str = "cpu",
    verbose: bool = True,
    checkpoint_path=None,
    save_every_epoch: bool = True,
    resume: bool = True,
):
    """
    `neg_per_pos`: if set (e.g. 10.0), training SEEDS are subsampled to
    keep all positives plus `neg_per_pos` negatives per positive (see
    `subsample_negatives`) -- dramatically fewer batches/epoch on a
    heavily-imbalanced graph. None (default) = use every training node,
    matching the original behavior.

    `checkpoint_path`: path to a single checkpoint FILE (e.g.
    `cache/checkpoints/latest_checkpoint.pt`), not a directory -- the
    caller picks the directory/filename (see the notebook, which builds
    this from CHECKPOINT_DIR). None (default) disables checkpointing
    entirely: nothing is saved, and `resume` is ignored.

    `save_every_epoch`: if True (default) and `checkpoint_path` is set,
    write a checkpoint after every epoch (model/optimizer/scheduler
    state + history), so a Colab disconnect loses at most one epoch.
    If False, checkpointing is skipped even though `checkpoint_path`
    is set -- e.g. for a quick local smoke-test run you don't want to
    clobber a real checkpoint with.

    `resume`: if True (default) and a checkpoint already exists at
    `checkpoint_path`, load model/optimizer/scheduler/history from it
    and continue from the next epoch instead of starting over. If
    False, any existing checkpoint at `checkpoint_path` is ignored
    (training starts fresh from epoch 1) -- it will still be
    overwritten by this run if `save_every_epoch` is also True.
    """
    model.to(device)
    train_idx, val_idx = chronological_split(data.num_nodes, val_frac)
    if neg_per_pos is not None:
        n_before = len(train_idx)
        train_idx = subsample_negatives(data.y, train_idx, neg_per_pos=neg_per_pos)
        if verbose:
            print(f"neg_per_pos={neg_per_pos}: training seeds {n_before:,} -> {len(train_idx):,} "
                  f"({n_before / max(len(train_idx), 1):.1f}x fewer batches/epoch)")

    train_loader = SimpleNeighborLoader(
        data, predecessors,
        num_neighbors=list(num_neighbors),
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = SimpleNeighborLoader(
        data, predecessors,
        num_neighbors=list(num_neighbors),
        input_nodes=val_idx,
        batch_size=batch_size,
        shuffle=False,
    )

    n_pos = data.y[train_idx].sum().item()
    n_neg = len(train_idx) - n_pos
    
    if neg_per_pos is not None:
        # If we already balanced the data via subsampling, DO NOT double-weight.
        pos_weight = torch.tensor([1.0], device=device)
        if verbose:
            print("Subsampling enabled. pos_weight set to 1.0 to avoid double-counting.")
    else:
        # If using all data, apply the natural class weight.
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        if verbose:
            print(f"Train pos_weight (class-imbalance correction): {pos_weight.item():.2f} "
                  f"({int(n_pos)} positive / {int(n_neg)} negative)")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}

    if resume and checkpoint_path is not None and checkpoint_path.exists():

        print(f"Resuming from {checkpoint_path}")

        ckpt = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            ckpt["model_state"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler_state"]
        )

        history = ckpt["history"]

        start_epoch = ckpt["epoch"] + 1

    t0_train = time.time()
    sample_time_total = 0.0
    compute_time_total = 0.0
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_losses = []
        t_sample_start = time.time()
        for batch in train_loader:
            sample_time_total += time.time() - t_sample_start

            t_compute_start = time.time()
            batch = batch.to(device)

            optimizer.zero_grad()
            logits = model(batch)[:batch.seed_size]
            y = batch.y[:batch.seed_size]
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            compute_time_total += time.time() - t_compute_start

            t_sample_start = time.time()

        val_loss, val_pr_auc = _evaluate_loader(model, val_loader, loss_fn, device)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        scheduler.step(val_pr_auc)

        if save_every_epoch and checkpoint_path is not None:
            torch.save(
                {
                    "epoch": epoch,

                    "model_state":
                        model.state_dict(),

                    "optimizer_state":
                        optimizer.state_dict(),

                    "scheduler_state":
                        scheduler.state_dict(),

                    "history":
                        history,
                },
                checkpoint_path,
            )

        if verbose:
            print(f"Epoch {epoch:>2d}/{num_epochs}  "
                    f"train_loss={history['train_loss'][-1]:.4f}  "
                    f"val_loss={val_loss:.4f}  val_PR-AUC={val_pr_auc:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

    print("05. Model Training (total): ", time.time() - t0_train)
    print("05a. Training - Neighbor Sampling (cumulative): ", sample_time_total)
    print("05b. Training - Model Forward/Backward (cumulative): ", compute_time_total)
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


def run_inference(model, data, predecessors, batch_size=1024, num_neighbors=(15, 10), device="cpu"):
    """Full-dataset scoring pass (PRD Section 14). Returns probs[N] aligned to data node order."""
    model.eval()
    all_idx = torch.arange(data.num_nodes)
    loader = SimpleNeighborLoader(
        data, predecessors,
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
