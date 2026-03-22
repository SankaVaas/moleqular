import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from typing import List
import copy


class DeepEnsemble:
    """
    Deep Ensemble for Uncertainty Quantification in Molecular Property Prediction.
    
    Theory (Lakshminarayanan et al., NeurIPS 2017):
    Train M independent models with different random seeds.
    The ensemble mean is the prediction, the ensemble variance
    captures EPISTEMIC uncertainty — uncertainty from lack of data.
    
    For a new molecule x:
        μ*(x) = (1/M) Σᵢ μᵢ(x)          — ensemble mean (prediction)
        σ²*(x) = (1/M) Σᵢ [σᵢ²(x) + μᵢ²(x)] - μ*²(x)  — total variance
    
    Key insight: if all ensemble members agree → low uncertainty (in-distribution)
                 if members disagree → high uncertainty (out-of-distribution)
    
    Why ensembles over Bayesian NNs?
    - Ensembles are better calibrated empirically (Ovadia et al., 2019)  
    - No approximation needed (vs variational inference)
    - Trivially parallelizable
    - Each member is a standard SchNet — no architecture changes needed
    """
    def __init__(self, models: List[nn.Module], normalizer=None):
        self.models = models
        self.normalizer = normalizer
        self.n_members = len(models)

    def predict(self, data: Data, device: str = 'cpu') -> dict:
        """
        Run ensemble prediction with uncertainty estimates.
        
        Returns:
            mean:     ensemble mean prediction (physical units)
            std:      ensemble std (epistemic uncertainty)
            all_preds: all member predictions
            cv:       coefficient of variation (relative uncertainty)
        """
        preds = []
        for model in self.models:
            model.eval()
            model.to(device)
            with torch.no_grad():
                data = data.to(device)
                # Add batch dimension if missing
                if not hasattr(data, 'batch') or data.batch is None:
                    data.batch = torch.zeros(
                        data.z.shape[0], dtype=torch.long, device=device)
                pred = model(data).cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)  # [M, B]

        # Ensemble statistics
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        # Denormalize if normalizer provided
        if self.normalizer:
            mean = mean * self.normalizer.std + self.normalizer.mean
            std = std * abs(self.normalizer.std)

        # Coefficient of variation — relative uncertainty
        cv = np.abs(std / (mean + 1e-8))

        return {
            "mean": mean,
            "std": std,
            "all_preds": preds,
            "cv": cv,
            "n_members": self.n_members
        }

    def predict_loader(self, loader: DataLoader, device: str = 'cpu') -> dict:
        """Run ensemble prediction over entire dataloader."""
        all_means, all_stds, all_targets = [], [], []

        for batch in loader:
            batch = batch.to(device)
            result = self.predict(batch, device=device)
            all_means.extend(result["mean"].flatten())
            all_stds.extend(result["std"].flatten())
            if hasattr(batch, 'y') and batch.y is not None:
                targets = batch.y.squeeze(-1).cpu().numpy()
                if self.normalizer:
                    targets = targets * self.normalizer.std + self.normalizer.mean
                all_targets.extend(targets)

        return {
            "mean": np.array(all_means),
            "std": np.array(all_stds),
            "targets": np.array(all_targets) if all_targets else None
        }

    @classmethod
    def from_checkpoints(cls, checkpoint_paths: List[str],
                         model_class, model_kwargs: dict, normalizer=None):
        """Load ensemble from saved checkpoints."""
        models = []
        for path in checkpoint_paths:
            ckpt = torch.load(path, map_location='cpu')
            model = model_class(**model_kwargs)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            models.append(model)
        print(f"Loaded ensemble of {len(models)} models")
        return cls(models, normalizer)


class MCDropoutModel(nn.Module):
    """
    MC Dropout for Uncertainty Quantification.
    
    Theory (Gal & Ghahramani, ICML 2016):
    Dropout at INFERENCE time approximates Bayesian inference.
    Each forward pass with different dropout masks = one posterior sample.
    
    Run T forward passes → T predictions → mean and variance.
    
    Epistemic uncertainty: σ²_epistemic = Var[f(x)]  (across T passes)
    
    Advantage over ensembles: single model, cheaper inference
    Disadvantage: underestimates uncertainty vs true Bayesian posterior
    """
    def __init__(self, base_model: nn.Module, dropout_p: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout_p = dropout_p
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout in all layers."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_p

    def enable_mc_dropout(self):
        """Set model to train mode to keep dropout active at inference."""
        self.base_model.train()

    def predict_with_uncertainty(self, data: Data, n_passes: int = 30,
                                  device: str = 'cpu') -> dict:
        """
        Run T stochastic forward passes.
        
        Args:
            n_passes: Number of MC samples (T). Higher = better estimate, slower.
                     30-100 is typical in literature.
        """
        self.enable_mc_dropout()
        self.base_model.to(device)
        data = data.to(device)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(
                data.z.shape[0], dtype=torch.long, device=device)

        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                pred = self.base_model(data).cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)  # [T, B]
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        # 95% confidence interval
        ci_95 = 1.96 * std

        return {
            "mean": mean,
            "std": std,
            "ci_95": ci_95,
            "all_preds": preds,
            "n_passes": n_passes
        }