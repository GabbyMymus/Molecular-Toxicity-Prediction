"""Training engine: masked BCE with per-endpoint class weights, early stopping on val PR AUC."""
from __future__ import annotations

import copy
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.dl.graph import labels_of

try:  # use the Phase 1 harness metric when the Phase 1 repo is on the path
    from src.evaluate import masked_metrics
except ImportError:  # standalone fallback; identical definition (contract rule 12)
    from sklearn.metrics import average_precision_score, roc_auc_score

    def masked_metrics(y_true, y_score):
        m = ~np.isnan(y_true)
        y, s = y_true[m], y_score[m]
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            return np.nan, np.nan
        return roc_auc_score(y, s), average_precision_score(y, s)

DEFAULTS = dict(lr=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=60, patience=10,
                grad_clip=5.0, use_pos_weight=True, pos_weight_cap=50.0, mask_missing=True,
                lr_patience=5, lr_factor=0.5)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(Y_train: np.ndarray, cap: Optional[float] = 50.0) -> torch.Tensor:
    """Per endpoint: n_neg / n_pos over the LABELED training rows (NaN ignored)."""
    pw = []
    for j in range(Y_train.shape[1]):
        col = Y_train[:, j]
        col = col[~np.isnan(col)]
        n_pos = max(float(col.sum()), 1.0)
        n_neg = float(len(col)) - float(col.sum())
        w = n_neg / n_pos
        pw.append(min(w, cap) if cap else w)
    return torch.tensor(pw, dtype=torch.float)


def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor,
                           pos_weight: Optional[torch.Tensor] = None,
                           mask_missing: bool = True) -> torch.Tensor:
    """Mean BCE over the labeled entries only. NaN labels contribute exactly zero.

    ``mask_missing=False`` is the deliberately WRONG variant (NaN treated as 0) used as an ablation.
    """
    valid = ~torch.isnan(y)
    if not mask_missing:
        valid = torch.ones_like(valid)
    y0 = torch.nan_to_num(y, nan=0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y0, pos_weight=pos_weight, reduction="none")
    loss = loss * valid.float()
    return loss.sum() / valid.float().sum().clamp(min=1.0)


@torch.no_grad()
def predict(model: nn.Module, graphs: Sequence, device, batch_size: int = 256) -> np.ndarray:
    """Probabilities (N, n_tasks) in the order of ``graphs``."""
    model.eval()
    out = []
    for batch in DataLoader(list(graphs), batch_size=batch_size, shuffle=False):
        out.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.vstack(out)


def evaluate(Y: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roc = np.full(Y.shape[1], np.nan)
    pr = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        roc[j], pr[j] = masked_metrics(Y[:, j], P[:, j])
    return roc, pr


def fit(model: nn.Module, train_graphs: Sequence, val_graphs: Sequence, seed: int = 0,
        device: str = "cpu", verbose: bool = False, **overrides) -> dict:
    """Train with early stopping on mean validation PR AUC. Restores the best weights.

    Returns dict(history=[...], best_epoch, best_val_pr, best_val_roc).
    """
    cfg = {**DEFAULTS, **overrides}
    set_seed(seed)
    model.to(device)
    Y_train = labels_of(train_graphs)
    Y_val = labels_of(val_graphs)
    pos_weight = compute_pos_weight(Y_train, cfg["pos_weight_cap"]).to(device) \
        if cfg["use_pos_weight"] else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=cfg["lr_factor"], patience=cfg["lr_patience"])
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(list(train_graphs), batch_size=cfg["batch_size"], shuffle=True,
                        generator=gen)
    best = dict(pr=-1.0, roc=np.nan, epoch=-1, state=None)
    history, bad = [], 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = masked_bce_with_logits(model(batch), batch.y, pos_weight, cfg["mask_missing"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            total += float(loss.detach()) * batch.num_graphs
            n += batch.num_graphs
        roc, pr = evaluate(Y_val, predict(model, val_graphs, device))
        val_pr, val_roc = float(np.nanmean(pr)), float(np.nanmean(roc))
        if np.isnan(val_pr):                       # no endpoint had both classes (tiny toy splits)
            val_pr, val_roc = 0.0, 0.5
        sched.step(val_pr)
        history.append(dict(epoch=epoch, train_loss=total / max(n, 1), val_pr=val_pr, val_roc=val_roc))
        if verbose:
            print(f"ep {epoch:3d} loss {total / n:.4f} val PR {val_pr:.4f} ROC {val_roc:.4f}")
        if val_pr > best["pr"] + 1e-5:
            best.update(pr=val_pr, roc=val_roc, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best["state"])
    return dict(history=history, best_epoch=best["epoch"], best_val_pr=best["pr"],
                best_val_roc=best["roc"])
