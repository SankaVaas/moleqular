import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
import time


class Normalizer:
    """
    Normalize targets to zero mean and unit variance.
    Critical for stable training — raw QM9 energies span thousands of meV.
    Store mean/std to denormalize predictions back to physical units.
    """
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean

    def denormalize_mae(self, mae: float) -> float:
        """Convert normalized MAE back to physical units."""
        return mae * abs(self.std)


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_epoch(model, loader, optimizer, normalizer, device):
    model.train()
    total_loss = 0
    total_mae = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Normalize targets
        y = normalizer.normalize(batch.y.squeeze(-1))

        # Forward pass
        pred = model(batch)

        # MSE loss for training
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()

        # Gradient clipping — important for molecular property prediction
        # Large gradients can destabilize training near energy discontinuities
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # MAE in normalized space
        with torch.no_grad():
            mae = nn.functional.l1_loss(pred, y)

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

    return total_loss / n_batches, total_mae / n_batches


@torch.no_grad()
def evaluate(model, loader, normalizer, device):
    model.eval()
    total_mae = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        y = normalizer.normalize(batch.y.squeeze(-1))
        pred = model(batch)

        mae = nn.functional.l1_loss(pred, y)
        total_mae += mae.item()
        n_batches += 1

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    avg_mae = total_mae / n_batches
    return avg_mae, np.array(all_preds), np.array(all_targets)


def train(model, train_data, val_data, config: dict, normalizer: Normalizer,
          save_dir: str = "./checkpoints"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0       # 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # Optimizer — AdamW with weight decay
    # Weight decay acts as L2 regularization, important for avoiding overfitting
    # on the smooth energy landscape
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler — reduce on plateau
    # Physical motivation: energy surfaces have flat regions near minima
    # where constant LR causes oscillation
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=10, min_lr=1e-6
    )

    early_stopping = EarlyStopping(patience=config.get("patience", 30))

    history = {
        "train_loss": [], "train_mae": [],
        "val_mae": [], "lr": [],
        "best_val_mae": float('inf'),
        "best_epoch": 0
    }

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train MAE':>12} "
          f"{'Val MAE':>12} {'Val MAE(phys)':>14} {'LR':>10}")
    print("-" * 72)

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, normalizer, device
        )
        val_mae, val_preds, val_targets = evaluate(
            model, val_loader, normalizer, device
        )

        # Denormalize MAE to physical units
        val_mae_phys = normalizer.denormalize_mae(val_mae)

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        # Save best model
        if val_mae < history["best_val_mae"]:
            history["best_val_mae"] = val_mae
            history["best_epoch"] = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_mae_phys": val_mae_phys,
                "config": config,
                "normalizer_mean": normalizer.mean,
                "normalizer_std": normalizer.std,
            }, f"{save_dir}/best_model.pt")

        elapsed = time.time() - start
        print(f"{epoch:>6} {train_loss:>12.6f} {train_mae:>12.6f} "
              f"{val_mae:>12.6f} {val_mae_phys:>12.4f} meV "
              f"{current_lr:>10.2e}  [{elapsed:.1f}s]")

        if early_stopping.step(val_mae):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save training history
    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Val MAE: {history['best_val_mae']:.6f} "
          f"({normalizer.denormalize_mae(history['best_val_mae']):.4f} meV) "
          f"at epoch {history['best_epoch']}")

    return history