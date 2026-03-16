"""
Quick local training run — small subset, CPU, just to verify everything works
before launching full training on Colab T4.
"""
import torch
from data.dataset import load_qm9
from models.schnet import SchNet
from training.trainer import train, Normalizer

# Small config for local CPU smoke test
config = {
    "hidden_dim": 64,
    "n_interactions": 2,
    "n_gaussians": 25,
    "cutoff": 5.0,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 10,           # just 3 epochs locally
    "patience": 10,
    "target_idx": 7,       # U0 — internal energy at 0K
}

print("Loading QM9 (small subset for smoke test)...")
train_data, val_data, test_data, mean, std = load_qm9(
    target_idx=config["target_idx"]
)

# Use tiny subset locally — full training on Colab
train_data = train_data[:500]
val_data = val_data[:100]

normalizer = Normalizer(mean=mean, std=std)

model = SchNet(
    hidden_dim=config["hidden_dim"],
    n_interactions=config["n_interactions"],
    n_gaussians=config["n_gaussians"],
    cutoff=config["cutoff"],
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
history = train(model, train_data, val_data, config, normalizer)