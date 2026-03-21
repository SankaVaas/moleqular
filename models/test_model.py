import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data, Batch
from models.schnet import SchNet

def test_equivariance():
    """
    THE most important test in this project.
    Verify that rotating a molecule gives the same energy prediction.
    If this fails, the model is physically wrong.
    """
    model = SchNet(hidden_dim=64, n_interactions=2)
    model.eval()

    # Create a fake 3-atom molecule
    z = torch.tensor([6, 1, 1])         # Carbon + 2 Hydrogens
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    # Build edges manually
    edge_index = torch.tensor([[0,1,0,2,1,0,2,0],[1,0,2,0,0,1,0,2]])
    diffs = pos[edge_index[0]] - pos[edge_index[1]]
    edge_attr = torch.norm(diffs, dim=-1, keepdim=True)
    batch = torch.zeros(3, dtype=torch.long)

    data = Data(z=z, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, batch=batch)

    # Predict original energy
    with torch.no_grad():
        E_original = model(data).item()

    # Rotate 90 degrees around z-axis
    R = torch.tensor([
        [0., -1., 0.],
        [1.,  0., 0.],
        [0.,  0., 1.]
    ])
    pos_rotated = pos @ R.T
    diffs_r = pos_rotated[edge_index[0]] - pos_rotated[edge_index[1]]
    edge_attr_r = torch.norm(diffs_r, dim=-1, keepdim=True)

    data_rotated = Data(z=z, pos=pos_rotated, edge_index=edge_index,
                        edge_attr=edge_attr_r, batch=batch)

    with torch.no_grad():
        E_rotated = model(data_rotated).item()

    print(f"Original energy:  {E_original:.6f}")
    print(f"Rotated energy:   {E_rotated:.6f}")
    print(f"Difference:       {abs(E_original - E_rotated):.2e}")

    assert abs(E_original - E_rotated) < 1e-4, \
        f"EQUIVARIANCE VIOLATED: {abs(E_original - E_rotated)}"
    print("✅ Equivariance test PASSED — model is physically correct")

def test_forward_pass():
    model = SchNet(hidden_dim=64, n_interactions=2)
    z = torch.tensor([6, 1, 1, 1, 1])
    pos = torch.randn(5, 3)
    edge_index = torch.tensor([[0,1,0,2,1,0],[1,0,2,0,0,1]])
    diffs = pos[edge_index[0]] - pos[edge_index[1]]
    edge_attr = torch.norm(diffs, dim=-1, keepdim=True)
    batch = torch.zeros(5, dtype=torch.long)
    data = Data(z=z, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, batch=batch)
    out = model(data)
    print(f"✅ Forward pass PASSED — output shape: {out.shape}")

if __name__ == "__main__":
    test_forward_pass()
    test_equivariance()