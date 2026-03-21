import torch
import numpy as np
from torch_geometric.datasets import QM9
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch_geometric.transforms as T

# QM9 target properties — we'll predict all, focus on a few
QM9_TARGETS = {
    0:  'mu',        # Dipole moment (Debye)
    1:  'alpha',     # Isotropic polarizability (Bohr^3)
    2:  'homo',      # HOMO energy (Hartree)
    3:  'lumo',      # LUMO energy (Hartree)
    4:  'gap',       # HOMO-LUMO gap (Hartree)
    5:  'r2',        # Electronic spatial extent (Bohr^2)
    6:  'zpve',      # Zero point vibrational energy (Hartree)
    7:  'u0',        # Internal energy at 0K (Hartree)
    8:  'u298',      # Internal energy at 298K (Hartree)
    9:  'h298',      # Enthalpy at 298K (Hartree)
    10: 'g298',      # Free energy at 298K (Hartree)
    11: 'cv',        # Heat capacity at 298K (cal/mol/K)
}

class MolecularGraphTransform(BaseTransform):
    """
    Transforms QM9 data into our equivariant-ready format.
    
    Key design decisions:
    - Use interatomic DISTANCES as edge features (rotationally invariant)
    - Store raw positions for force computation
    - Add fully-connected edges within cutoff radius
    """
    def __init__(self, cutoff: float = 5.0):
        self.cutoff = cutoff  # Angstroms
    
    def forward(self, data: Data) -> Data:
        pos = data.pos          # [N, 3] atomic positions
        z = data.z              # [N] atomic numbers
        N = pos.shape[0]
        
        # Build edges within cutoff (fully connected for small molecules)
        edge_index = []
        edge_attr = []
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pos[i] - pos[j]
                dist = torch.norm(diff).item()
                if dist < self.cutoff:
                    edge_index.append([i, j])
                    edge_attr.append(dist)
        
        if len(edge_index) == 0:
            # Fallback: connect all atoms
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edge_index.append([i, j])
                        diff = pos[i] - pos[j]
                        edge_attr.append(torch.norm(diff).item())
        
        data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
        data.pos = pos          # Keep positions for equivariant layers
        data.z = z              # Atomic numbers as node features
        
        return data

def load_qm9(root: str = './data/raw', target_idx: int = 7, 
             cutoff: float = 5.0, split: tuple = (0.8, 0.1, 0.1)):
    """
    Load QM9 dataset with our molecular graph transform.
    
    Args:
        target_idx: Which property to predict (7 = U0, internal energy at 0K)
        cutoff: Distance cutoff for edges in Angstroms
        split: Train/val/test split ratios
    """
    transform = MolecularGraphTransform(cutoff=cutoff)
    dataset = QM9(root=root, transform=transform)
    
    # Use target property
    dataset.data.y = dataset.data.y[:, target_idx:target_idx+1]
    
    # Compute mean and std for normalization
    y = dataset.data.y
    mean = y.mean().item()
    std = y.std().item()
    print(f"Target: {QM9_TARGETS[target_idx]}")
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"Dataset size: {len(dataset)}")
    
    # Split
    N = len(dataset)
    n_train = int(N * split[0])
    n_val = int(N * split[1])
    
    train = dataset[:n_train]
    val = dataset[n_train:n_train + n_val]
    test = dataset[n_train + n_val:]
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test, mean, std

if __name__ == "__main__":
    train, val, test, mean, std = load_qm9()
    sample = train[0]
    print(f"\nSample molecule:")
    print(f"  Atoms: {sample.z.shape[0]}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Positions shape: {sample.pos.shape}")
    print(f"  Edge distances shape: {sample.edge_attr.shape}")
    print(f"  Target: {sample.y.item():.4f}")