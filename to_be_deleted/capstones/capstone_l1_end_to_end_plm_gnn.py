"""
Capstone 1: End-to-End Fine-Tuned pLM + GNN
============================================
What you'll learn
-----------------
- How to jointly train a protein language model (ESM-2) and a graph neural
  network (GAT) so that gradients from the task loss flow through BOTH.
- Why end-to-end training can lift accuracy over a frozen-pLM baseline —
  and what the memory/compute cost looks like.
- How to set DIFFERENTIAL LEARNING RATES: a small LR for the pLM body
  (which is already well-trained) and a larger LR for the freshly-
  initialised GNN head.
- How to count trainable parameters and compare model configurations.

How this differs from gnn_l4
-----------------------------
In gnn_l4 the pLM was FROZEN: embeddings were pre-computed once, cached,
and the GNN trained on static node features. Here the pLM forward pass
lives INSIDE the training graph, so the pLM weights are updated alongside
the GNN weights on every step. This is "end-to-end" fine-tuning.

Memory note
-----------
Because ESM-2 activations must be stored for backprop, peak GPU memory is
roughly 4-8x higher than the frozen baseline. To stay CPU-friendly we use
tiny dataset slices (N_TRAIN~120, N_TEST~40), short sequences (MAX_LEN~200),
and a batch size of 2. Scale these up if you have a GPU with >=8 GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLM_NAME = "facebook/esm2_t6_8M_UR50D"   # 8 M-param ESM-2; change to t12_35M for more capacity
DATASET_NAME = "zhanglab/DeepSol"
N_TRAIN = 120        # keep small — full pLM backprop is expensive
N_TEST = 40
MAX_LEN = 200        # truncate sequences to this length
WINDOW = 3           # sequence-window graph: each residue connects to ±WINDOW neighbours
HIDDEN = 64          # GAT hidden dim
HEADS = 4            # GAT attention heads
EPOCHS = 5           # few epochs for demo; increase with GPU
BATCH_SIZE = 2       # micro-batch because we keep the pLM in the compute graph

# Differential learning rates: pLM body is already pre-trained, so it needs
# a much smaller nudge than the randomly-initialised GNN head.
LR_PLM = 5e-5        # LR for ESM-2 weights
LR_GNN = 5e-4        # LR for GAT + classifier weights (10x larger)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def sequence_window_graph(seq_len: int, window: int = WINDOW) -> torch.Tensor:
    """Return edge_index for a sequence-window graph of length seq_len.

    Every residue i is connected to residues i±1, i±2, ..., i±window.
    This is a simple local-connectivity prior: nearby residues in sequence
    are likely in contact in 3-D (especially for short-range secondary
    structure). See gnn_l6 for a real contact-map alternative.
    """
    edges = []
    for i in range(seq_len):
        for d in range(1, window + 1):
            if i + d < seq_len:
                edges.append((i, i + d))
                edges.append((i + d, i))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------------------
# End-to-end model: ESM-2 body + GAT head in a single nn.Module
# ---------------------------------------------------------------------------

class EndToEndPLMGNN(nn.Module):
    """A single nn.Module that owns BOTH the pLM encoder and the GNN classifier.

    forward() accepts a list of (already-tokenised) input dicts — one per
    sequence in the logical batch — runs ESM-2 on each, strips the special
    tokens to get per-residue embeddings, assembles a PyG Batch, and passes
    it through the GAT layers to produce one logit vector per sequence.

    Why loop over sequences rather than padding into a true batch?
    Each sequence has a different length, so the derived graph has a different
    number of nodes. PyG's Batch object handles heterogeneous graphs cleanly;
    padding the pLM inputs and masking out graph nodes is trickier. The loop
    is slightly slower but much easier to understand and debug.
    """

    def __init__(self, plm_name: str, hidden: int, out_channels: int, heads: int):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name)
        plm_dim = self.plm.config.hidden_size

        # GAT layer 1: plm_dim -> hidden*heads
        self.gat1 = GATConv(plm_dim, hidden, heads=heads, dropout=0.2)
        # GAT layer 2: hidden*heads -> hidden (single head, averaged)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.2)
        self.classifier = nn.Linear(hidden, out_channels)

    def forward(self, token_batches: list, seq_lens: list) -> torch.Tensor:
        """
        Args:
            token_batches: list of dicts, each the output of tokenizer(seq, ...)
                           and already moved to the correct device.
            seq_lens:      list of ints — the TRUE (post-truncation) sequence
                           lengths, used to strip <cls>/<eos> special tokens.

        Returns:
            logits: (N, out_channels) tensor, one row per sequence.
        """
        data_list = []
        for inputs, seq_len in zip(token_batches, seq_lens):
            # ESM-2 forward — NO torch.no_grad(), so gradients flow back here.
            hidden = self.plm(**inputs).last_hidden_state   # (1, L+2, D)
            # Slice away <cls> (index 0) and <eos> (index -1 or beyond seq_len)
            node_feats = hidden[0, 1 : 1 + seq_len]         # (seq_len, D)

            edge_index = sequence_window_graph(seq_len, WINDOW).to(node_feats.device)
            data_list.append(Data(x=node_feats, edge_index=edge_index))

        # Merge individual sequence graphs into one disconnected PyG Batch.
        # global_mean_pool later uses batch.batch to pool each graph separately.
        pyg_batch = Batch.from_data_list(data_list)

        h = F.elu(self.gat1(pyg_batch.x, pyg_batch.edge_index))
        h = F.elu(self.gat2(h, pyg_batch.edge_index))
        graph_emb = global_mean_pool(h, pyg_batch.batch)    # (N, hidden)
        return self.classifier(graph_emb)                   # (N, out_channels)


# ---------------------------------------------------------------------------
# Frozen-pLM baseline — identical architecture, pLM locked
# ---------------------------------------------------------------------------

class FrozenPLMGNN(EndToEndPLMGNN):
    """Same architecture as EndToEndPLMGNN but with the pLM frozen.

    This is the gnn_l4 approach expressed as an nn.Module for a fair
    apples-to-apples comparison: same EPOCHS, same data, same GNN capacity,
    only the gradient flow differs.
    """

    def __init__(self, plm_name: str, hidden: int, out_channels: int, heads: int):
        super().__init__(plm_name, hidden, out_channels, heads)
        # Freeze every pLM parameter so the optimiser ignores them.
        for p in self.plm.parameters():
            p.requires_grad = False
        self.plm.eval()   # also disables dropout inside the pLM

    def forward(self, token_batches: list, seq_lens: list) -> torch.Tensor:
        # Wrap the pLM call in no_grad to avoid storing activations for backprop —
        # this is the memory saving that makes the frozen baseline much cheaper.
        data_list = []
        for inputs, seq_len in zip(token_batches, seq_lens):
            with torch.no_grad():
                hidden = self.plm(**inputs).last_hidden_state
            node_feats = hidden[0, 1 : 1 + seq_len]
            edge_index = sequence_window_graph(seq_len, WINDOW).to(node_feats.device)
            data_list.append(Data(x=node_feats, edge_index=edge_index))

        pyg_batch = Batch.from_data_list(data_list)
        h = F.elu(self.gat1(pyg_batch.x, pyg_batch.edge_index))
        h = F.elu(self.gat2(h, pyg_batch.edge_index))
        return self.classifier(global_mean_pool(h, pyg_batch.batch))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(raw_split, tokenizer, device):
    """Tokenise each sequence and return (token_inputs, seq_len, label) triples.

    We tokenise eagerly here so the training loop only has to move pre-built
    tensors to the device on each step, rather than re-tokenising every time.
    """
    samples = []
    for item in raw_split:
        seq = item["sequence"][:MAX_LEN]
        label = item["label"]
        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN + 2,   # +2 for <cls> and <eos>
        )
        # Move to device once; keep on CPU until training loop moves it.
        inputs = {k: v.to(device) for k, v in inputs.items()}
        samples.append((inputs, len(seq), label))
    return samples


def make_batches(samples, batch_size):
    """Yield list-of-samples chunks of size batch_size."""
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, samples, optimizer, device):
    model.train()
    # FrozenPLMGNN overrides train() via super(), but we want the pLM to stay
    # in eval mode (no dropout) even during training — enforce that here.
    if isinstance(model, FrozenPLMGNN):
        model.plm.eval()

    total_loss = 0.0
    for chunk in make_batches(samples, BATCH_SIZE):
        token_batches = [s[0] for s in chunk]
        seq_lens = [s[1] for s in chunk]
        labels = torch.tensor([s[2] for s in chunk], dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(token_batches, seq_lens)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(chunk)
    return total_loss / len(samples)


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    correct = 0
    for chunk in make_batches(samples, BATCH_SIZE):
        token_batches = [s[0] for s in chunk]
        seq_lens = [s[1] for s in chunk]
        labels = torch.tensor([s[2] for s in chunk], dtype=torch.long, device=device)
        logits = model(token_batches, seq_lens)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
    return correct / len(samples)


def build_optimizer(model: nn.Module, is_end_to_end: bool):
    """Build an Adam optimiser with differential learning rates.

    Parameter groups let you assign different hyperparameters to different
    parts of the model. Here:
      - pLM body  => small LR (pre-trained; large updates would destroy it)
      - GNN head  => larger LR (randomly initialised; needs to move fast)

    For the frozen baseline the pLM group has requires_grad=False for every
    param, so PyTorch simply skips those in the update step. We still separate
    the groups for symmetry and to show the pattern clearly.
    """
    plm_params = list(model.plm.parameters())
    gnn_params = (
        list(model.gat1.parameters())
        + list(model.gat2.parameters())
        + list(model.classifier.parameters())
    )
    return torch.optim.Adam(
        [
            {"params": plm_params, "lr": LR_PLM},
            {"params": gnn_params, "lr": LR_GNN},
        ],
        weight_decay=1e-4,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 1. Load tokenizer (shared by both models) -----------------------
    print(f"\nLoading tokenizer: {PLM_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(PLM_NAME)

    # ---- 2. Load and subset dataset -------------------------------------
    print(f"Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    train_raw = raw["train"].select(range(N_TRAIN))
    test_raw = raw["test"].select(range(N_TEST))

    # ---- 3. Tokenise data -----------------------------------------------
    print("Tokenising sequences...")
    train_data = prepare_data(train_raw, tokenizer, device)
    test_data = prepare_data(test_raw, tokenizer, device)

    # ---- 4. Build and compare the two models ----------------------------
    print("\nBuilding models...")
    e2e_model = EndToEndPLMGNN(PLM_NAME, HIDDEN, 2, HEADS).to(device)
    frozen_model = FrozenPLMGNN(PLM_NAME, HIDDEN, 2, HEADS).to(device)

    e2e_params = count_trainable(e2e_model)
    frozen_params = count_trainable(frozen_model)
    print(f"  End-to-end trainable params : {e2e_params:,}")
    print(f"  Frozen-pLM trainable params : {frozen_params:,}")
    print(f"  pLM body params (frozen)    : {e2e_params - frozen_params:,}")

    e2e_opt = build_optimizer(e2e_model, is_end_to_end=True)
    frozen_opt = build_optimizer(frozen_model, is_end_to_end=False)

    # ---- 5. Train both models for the same epoch budget -----------------
    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}, "
          f"N_train={N_TRAIN}, MAX_LEN={MAX_LEN})...")
    print(f"{'epoch':>5}  {'e2e_loss':>9}  {'e2e_acc':>8}  "
          f"{'frozen_loss':>11}  {'frozen_acc':>10}")
    print("-" * 55)

    for ep in range(1, EPOCHS + 1):
        e2e_loss = train_one_epoch(e2e_model, train_data, e2e_opt, device)
        frozen_loss = train_one_epoch(frozen_model, train_data, frozen_opt, device)
        e2e_acc = evaluate(e2e_model, test_data, device)
        frozen_acc = evaluate(frozen_model, test_data, device)
        print(f"{ep:>5}  {e2e_loss:>9.4f}  {e2e_acc:>8.3f}  "
              f"{frozen_loss:>11.4f}  {frozen_acc:>10.3f}")

    # ---- 6. Final comparison summary ------------------------------------
    final_e2e = evaluate(e2e_model, test_data, device)
    final_frozen = evaluate(frozen_model, test_data, device)
    print("\n--- Final test accuracies ---")
    print(f"  End-to-end (pLM + GNN jointly trained) : {final_e2e:.3f}")
    print(f"  Frozen-pLM baseline (GNN head only)    : {final_frozen:.3f}")
    lift = final_e2e - final_frozen
    direction = "improvement" if lift >= 0 else "regression"
    print(f"  End-to-end {direction}: {lift:+.3f}")
    print(
        "\nNote: on a tiny N the lift may be noisy or even negative. "
        "Scale N_TRAIN / EPOCHS up on a GPU to see the real effect."
    )

    print(
        """
Things to experiment with:
- Unfreeze only the top-K ESM-2 layers (the rest stay frozen): iterate over
  model.plm.encoder.layer[-K:] and set p.requires_grad = True. Lower memory
  cost than full fine-tuning, often >90% of the accuracy gain.
- Add LoRA to the pLM for parameter-efficient fine-tuning (ties to plm_l7):
      pip install peft
      from peft import LoraConfig, get_peft_model
      model.plm = get_peft_model(model.plm, LoraConfig(r=8, lora_alpha=16))
- Use gradient checkpointing to trade compute for memory, enabling longer
  sequences or larger pLMs on the same GPU:
      model.plm.gradient_checkpointing_enable()
- Replace the sequence-window graph with a real predicted contact graph
  (gnn_l6) — the structural prior should help the GAT learn faster.
- Add a cosine LR schedule with linear warmup:
      from transformers import get_cosine_schedule_with_warmup
- Enable mixed precision (torch.autocast) to roughly halve memory and speed
  up GPU training: wrap the forward pass in `with torch.autocast("cuda"):`.
"""
    )


if __name__ == "__main__":
    main()
