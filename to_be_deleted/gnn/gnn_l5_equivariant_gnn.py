"""
GNN Lesson 5: Equivariant Graph Neural Networks
================================================
What you'll learn
-----------------
- Why "naive" GNNs LOSE 3D geometric information.
- Equivariance vs. invariance: the model's output transforms predictably
  when the input rotates / translates.
- Implement a minimal EGNN (Satorras et al. 2021) message-passing layer
  FROM SCRATCH so you can see exactly what makes it work.
- Verify equivariance numerically: rotate the input, check the output rotates
  the same way.

The intuition
-------------
A regular GCN treats each node as a bag of scalar features. If you rotate
your protein, the input features stay the same (one-hot AA doesn't rotate),
but the GEOMETRY changes — and a GCN can't see geometry at all because it
never received coordinates as input.

For 3D molecular tasks (force prediction, structure prediction, denoising)
we need a model that:
  - takes 3D coordinates as input,
  - is INVARIANT to rotation / translation for scalar predictions
    (energies, distances, classification),
  - is EQUIVARIANT for vector predictions
    (forces, predicted positions: rotate input -> rotate output).

The EGNN trick
--------------
Per-edge message:
    m_ij = phi_e( [h_i, h_j, ||x_i - x_j||^2] )

Coordinate update (EQUIVARIANT — uses VECTOR DIFFERENCES):
    x_i' = x_i + sum_j (x_i - x_j) * phi_x( m_ij )

Feature update (INVARIANT — only uses scalars):
    h_i' = phi_h( [h_i, sum_j m_ij] )

Why this is equivariant: the only geometric quantity ever fed to the
neural networks (`phi_*`) is `||x_i - x_j||^2`, which is rotation- and
translation-invariant. The coordinate update uses (x_i - x_j) — a VECTOR
that rotates the same way as the inputs. So if you rotate all `x_i`, the
output coords rotate the same way, and the scalar features `h_i` are
unchanged.

This lesson focuses on the MECHANICS — we'll demonstrate equivariance
numerically. Real applications use bigger versions of the same recipe
(DimeNet, NequIP, MACE, EquiFormer, ProteinMPNN).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# A minimal EGNN message-passing layer
# ---------------------------------------------------------------------------

class EGNNLayer(nn.Module):
    """One layer of EGNN — updates BOTH features `h` AND coordinates `x`.

    For clarity this is a dense (every-pair) implementation suitable for
    small graphs (a typical protein domain). For large graphs you'd use
    edge_index / scatter as in PyG.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # phi_e: builds a per-edge message from [h_i, h_j, ||x_i - x_j||^2]
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # phi_x: produces a SCALAR coefficient for each (i,j) pair, which
        # multiplies the vector (x_i - x_j) in the coordinate update.
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # phi_h: combines a node's own feature with the aggregated messages.
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x):
        """h: (N, hidden_dim) node features.  x: (N, 3) coordinates."""
        N = h.shape[0]

        # Pairwise vector differences and squared distances.
        # diffs[i, j] = x[j] - x[i].
        diffs = x.unsqueeze(0) - x.unsqueeze(1)            # (N, N, 3)
        sq_dist = (diffs * diffs).sum(dim=-1, keepdim=True)  # (N, N, 1)

        # Per-edge concat [h_i, h_j, ||x_i - x_j||^2].
        h_i = h.unsqueeze(1).expand(N, N, -1)              # (N, N, D)
        h_j = h.unsqueeze(0).expand(N, N, -1)              # (N, N, D)
        edge_input = torch.cat([h_i, h_j, sq_dist], dim=-1)  # (N, N, 2D+1)

        # Compute per-edge messages.
        m = self.phi_e(edge_input)                         # (N, N, D)

        # Mask out the diagonal (i == j has no real edge).
        eye = torch.eye(N, device=h.device).unsqueeze(-1)
        m = m * (1.0 - eye)

        # Coordinate update: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        # diffs[i, j] = x[j] - x[i], so (x_i - x_j) = -diffs[i, j].
        coord_coef = self.phi_x(m)                         # (N, N, 1)
        x_update = -(diffs * coord_coef).sum(dim=1)        # (N, 3)
        x_new = x + x_update / max(N - 1, 1)               # average for stability

        # Feature update: h_i' = phi_h([h_i, sum_j m_ij])
        m_agg = m.sum(dim=1) / max(N - 1, 1)               # (N, D)
        h_new = self.phi_h(torch.cat([h, m_agg], dim=-1))  # (N, D)

        return h_new, x_new


class EGNN(nn.Module):
    """Stack of EGNN layers."""

    def __init__(self, in_channels, hidden_dim, num_layers=3):
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x_feat, x_coord):
        h = self.embed(x_feat)
        for layer in self.layers:
            h, x_coord = layer(h, x_coord)
        return h, x_coord


# ---------------------------------------------------------------------------
# Numerical equivariance check
# ---------------------------------------------------------------------------

def random_rotation_matrix(seed=0):
    """A random 3x3 rotation matrix via QR decomposition of a random matrix."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    # Ensure determinant +1 (proper rotation, not reflection).
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def check_equivariance(model, x_feat, x_coord, atol=None):
    """Verify that rotating the input rotates the output coords the same way,
    and that scalar features `h` are unchanged.

    Note on tolerance: a CORRECT implementation can still produce visible
    error in fp32 because:
      - The rotation matrix R from QR decomposition is only orthogonal to
        ~1e-7 in fp32, so R @ R.T is not exactly the identity.
      - Stacking N layers compounds this error roughly as O(epsilon * N).
    With 3 EGNN layers and fp32, expect errors around 1e-2 to 1e-3 on
    coordinates. Switch the model to .double() and you'll see ~1e-14.
    """
    if atol is None:
        atol = 1e-10 if x_coord.dtype == torch.float64 else 5e-2

    R = random_rotation_matrix(seed=42).to(x_coord.dtype)
    t = torch.tensor([3.0, -2.0, 1.0], dtype=x_coord.dtype)

    # Run on original coords.
    h_a, x_a = model(x_feat, x_coord)

    # Run on rotated + translated coords.
    x_rt = x_coord @ R.T + t
    h_b, x_b = model(x_feat, x_rt)

    # Apply the same rotation+translation to the model's predicted coords.
    x_a_rt = x_a @ R.T + t

    feat_diff = (h_a - h_b).abs().max().item()
    coord_diff = (x_a_rt - x_b).abs().max().item()

    print(f"\nEquivariance check (precision = {x_coord.dtype}):")
    print(f"  max |h(x) - h(R x + t)|             = {feat_diff:.2e}")
    print(f"      (features should be INVARIANT — value unchanged by rotation/translation)")
    print(f"  max |R out(x) + t - out(R x + t)|   = {coord_diff:.2e}")
    print(f"      (coords should be EQUIVARIANT — they rotate/translate with the input)")

    if feat_diff < atol and coord_diff < atol:
        print(f"  PASSED — model is equivariant within atol={atol:.0e}.")
    else:
        print(f"  FAILED at atol={atol:.0e}. If running fp32, try fp64 to confirm "
              f"this is just numerical noise, not a bug.")


# ---------------------------------------------------------------------------
# Demo on a synthetic protein-like graph
# ---------------------------------------------------------------------------

def synthetic_helix_coords(n_residues=20, rise=1.5, radius=2.3, turn_deg=100.0, seed=0):
    """A noisy alpha-helix to act as our toy 'protein'."""
    g = torch.Generator().manual_seed(seed)
    angles = torch.deg2rad(torch.tensor(turn_deg) * torch.arange(n_residues))
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z = rise * torch.arange(n_residues).float()
    coords = torch.stack([x, y, z], dim=-1)
    # Add a little noise so it's not a perfect helix.
    coords = coords + 0.1 * torch.randn(*coords.shape, generator=g)
    return coords


def main():
    torch.manual_seed(0)
    n_residues = 20
    feat_dim = 8

    # Random per-residue features (would normally be one-hot AA or pLM embeddings).
    h_in = torch.randn(n_residues, feat_dim)
    x_in = synthetic_helix_coords(n_residues)

    print(f"Synthetic protein: {n_residues} residues")
    print(f"  feature dim: {feat_dim}")
    print(f"  coord shape: {tuple(x_in.shape)}")

    model = EGNN(in_channels=feat_dim, hidden_dim=16, num_layers=3)
    model.eval()

    # Forward pass.
    h_out, x_out = model(h_in, x_in)
    print(f"\nOutput shapes: h={tuple(h_out.shape)}, x={tuple(x_out.shape)}")
    print(f"Output coords (first 3 rows):\n{x_out[:3]}")

    # The headline experiment — first in fp32 (default), then in fp64 to
    # demonstrate that the residual error is just numerical precision.
    check_equivariance(model, h_in, x_in)

    print("\nNow repeat the check in float64 to isolate fp32 noise from real bugs:")
    model_d = model.double()
    check_equivariance(model_d, h_in.double(), x_in.double())

    print(
        """
Things to experiment with:
- Train it on something! Take a real structure, add Gaussian noise to its
  coords, and ask the EGNN to recover them (a learned denoiser). The MSE
  between predicted and ground-truth coords is a rotation-invariant loss.
- Use the equivariant features to predict per-residue forces (vectors that
  must rotate with the input).
- For larger systems, switch to PyG's MessagePassing base class with
  edge_index for sparsity.
- Read about more powerful equivariant models: NequIP, MACE, EquiFormer,
  Allegro, ProteinMPNN — all build on this same insight.
- Compare to TFN / SE(3)-Transformer: they use spherical-harmonic-based
  irreducible representations for higher-order equivariance.
"""
    )


if __name__ == "__main__":
    main()
