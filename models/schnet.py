import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data
import numpy as np


class GaussianSmearing(nn.Module):
    """
    Expand scalar distances into a continuous-filter basis.
    
    Physics motivation: distances alone are too coarse — we need to capture
    the smooth variation of interactions with distance. Gaussian basis functions
    centered at different distances give the model a rich distance representation.
    
    This is rotationally invariant by construction since ||rᵢ - rⱼ|| is invariant.
    """
    def __init__(self, start: float = 0.0, stop: float = 5.0, n_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / ((stop - start) / n_gaussians) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # Ensure dist is [E] not [E, 1]
        if dist.dim() == 2:
            dist = dist.squeeze(-1)
        dist = dist.unsqueeze(-1) - self.offset.unsqueeze(0)  # [E, n_gaussians]
        return torch.exp(self.coeff * dist ** 2)    


class ShiftedSoftplus(nn.Module):
    """
    Smooth activation function from SchNet paper.
    SSP(x) = ln(0.5·eˣ + 0.5) — smooth everywhere, no dead neurons
    Preferred over ReLU for molecular property prediction (smoother energy surfaces)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - np.log(2.0)


class InteractionBlock(nn.Module):
    """
    Core SchNet interaction: atom-wise + continuous filter convolution.
    
    Mathematical operation:
        v_i^{l+1} = Σⱼ ∈ N(i) h_j^l ⊙ W(eᵢⱼ)
    
    Where:
        h_j^l  = atom j's hidden state at layer l
        W(eᵢⱼ) = filter network applied to edge (distance) features
        ⊙       = element-wise product
    
    This is equivariant because eᵢⱼ = ||rᵢ - rⱼ|| is rotationally invariant,
    so W(eᵢⱼ) is invariant, and the message passing aggregation is permutation
    invariant. The whole operation respects SE(3) symmetry.
    """
    def __init__(self, hidden_dim: int, n_gaussians: int, cutoff: float):
        super().__init__()
        self.act = ShiftedSoftplus()

        # Filter network: distance features → filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Atom-wise transformations
        self.atom_wise_1 = nn.Linear(hidden_dim, hidden_dim)
        self.atom_wise_2 = nn.Linear(hidden_dim, hidden_dim)
        self.atom_wise_3 = nn.Linear(hidden_dim, hidden_dim)

        # Cosine cutoff for smooth distance decay
        self.cutoff = cutoff

    def cosine_cutoff(self, dist: torch.Tensor) -> torch.Tensor:
        """Smooth envelope that decays interactions to zero at cutoff."""
        return 0.5 * (torch.cos(dist * np.pi / self.cutoff) + 1.0) * (dist < self.cutoff).float()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_dist: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:

        row = edge_index[0]  # target atoms
        col = edge_index[1]  # source atoms

        # Ensure shapes are correct
        if edge_dist.dim() == 2:
            edge_dist = edge_dist.squeeze(-1)   # [E, 1] -> [E]
        if edge_rbf.dim() == 3:
            edge_rbf = edge_rbf.squeeze(1)      # [E, 1, G] -> [E, G]

        # Transform source atom features
        h_j = self.atom_wise_1(h[col])          # [E, hidden_dim]

        # Compute continuous filters from distances
        W = self.filter_net(edge_rbf)            # [E, hidden_dim]

        # Apply cosine cutoff envelope
        cutoff_vals = self.cosine_cutoff(edge_dist).unsqueeze(-1)  # [E, 1]
        W = W * cutoff_vals                      # [E, hidden_dim]

        # Element-wise product
        messages = h_j * W                       # [E, hidden_dim]

        # Aggregate messages at target atoms
        N = h.shape[0]
        agg = torch.zeros(N, h.shape[1], device=h.device)
        agg.scatter_add_(0, row.unsqueeze(1).expand_as(messages), messages)

        # Atom-wise update
        agg = self.atom_wise_2(agg)
        agg = self.act(agg)
        agg = self.atom_wise_3(agg)

        return h + agg


class SchNet(nn.Module):
    """
    SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions. (Schütt et al., 2017)
    
    Key properties:
    - SE(3) invariant by construction (uses only distances, not coordinates)
    - Energy-extensive: total energy = sum of atomic contributions
    - Smooth: uses Gaussian basis and ShiftedSoftplus activations
    
    Architecture:
        Atomic embeddings → L interaction blocks → Atom-wise readout → Sum pooling
    """
    def __init__(self,
                 hidden_dim: int = 128,
                 n_interactions: int = 3,
                 n_gaussians: int = 50,
                 cutoff: float = 5.0,
                 max_z: int = 100,
                 dropout: float = 0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Atom type embedding: atomic number → hidden vector
        # This is where chemical identity is encoded
        self.atom_embedding = nn.Embedding(max_z, hidden_dim, padding_idx=0)

        # Distance expansion: scalar distance → basis functions
        self.distance_expansion = GaussianSmearing(0.0, cutoff, n_gaussians)

        # Stack of interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, n_gaussians, cutoff)
            for _ in range(n_interactions)
        ])

        # Readout: per-atom energy contributions
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.act = ShiftedSoftplus()
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object with z, pos, edge_index, edge_attr, batch
        Returns:
            pred: [B] predicted molecular property
        """
        z = data.z                      # [N] atomic numbers
        edge_index = data.edge_index    # [2, E]
        edge_dist = data.edge_attr      # [E, 1] distances
        batch = data.batch              # [N] molecule index per atom

        # Initial atom embeddings from atomic numbers
        h = self.atom_embedding(z)      # [N, hidden_dim]
        # Expand distances to Gaussian basis
        edge_dist_squeezed = edge_dist.squeeze(-1) if edge_dist.dim() == 2 else edge_dist
        edge_rbf = self.distance_expansion(edge_dist_squeezed)  # [E, n_gaussians]

        # Apply interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_dist_squeezed, edge_rbf)

    
        # Per-atom energy contributions
        atom_energies = self.readout(h).squeeze(-1)  # [N]

        # Sum atomic contributions per molecule (energy-extensive)
        mol_energy = global_add_pool(
            atom_energies.unsqueeze(-1), batch
        ).squeeze(-1)                   # [B]

        return mol_energy

    def get_representations(self, data: Data) -> torch.Tensor:
        """Return atom-level representations for analysis."""
        z = data.z
        edge_index = data.edge_index
        edge_dist = data.edge_attr
        h = self.atom_embedding(z)
        edge_rbf = self.distance_expansion(edge_dist)
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_dist, edge_rbf)
        return h