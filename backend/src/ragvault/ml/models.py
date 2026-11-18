"""PyTorch models shared by training scripts and the runtime classifier."""

from __future__ import annotations

import torch
from torch import nn


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EmbeddingMLP(nn.Module):
    """Small MLP head on top of frozen sentence embeddings."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(x_train, y_train, x_val, y_val, n_classes: int, epochs: int = 40, lr: float = 1e-3,
              batch_size: int = 128, weight_decay: float = 1e-4, seed: int = 0, verbose: bool = False):
    """Train an EmbeddingMLP with early stopping on validation accuracy. Returns (model_on_cpu, history)."""
    torch.manual_seed(seed)
    device = best_device()
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    xv = torch.as_tensor(x_val, dtype=torch.float32).to(device)
    yv = torch.as_tensor(y_val, dtype=torch.long).to(device)
    model = EmbeddingMLP(xt.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, history, patience = -1.0, None, [], 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(1) == yv).float().mean().item()
        history.append({"epoch": epoch, "loss": total / len(xt), "val_acc": acc})
        if verbose:
            print(f"  epoch {epoch:02d} loss={total / len(xt):.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, patience = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), history
