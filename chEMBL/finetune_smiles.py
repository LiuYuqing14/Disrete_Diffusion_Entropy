"""
finetune_smiles.py
------------------
Fine-tune a pretrained SEDD model on ChEMBL molecular SMILES strings.

Key differences vs protein fine-tuning
---------------------------------------
1. Vocabulary: SMILES alphabet (~94 tokens) vs amino acids (25).
   Two-character atoms (Cl, Br, @@, [nH], ...) require a greedy tokeniser.

2. Length distribution: ChEMBL molecules are typically 10–120 tokens.
   We DROP sequences longer than max_length (not truncate), because
   a partial SMILES string is chemically meaningless.

3. Graph choice: both absorb and uniform work for SMILES.
   'absorb' is recommended — it mirrors the BERT-style masking that
   chemical language models (ChemBERTa, MolBERT) use successfully.

4. Evaluation: unlike proteins (Spearman ρ), molecules are evaluated on
   generation quality: validity, uniqueness, novelty, drug-likeness (QED).
   Run evaluate_smiles.py after training.

Usage
-----
  # From inside the SEDD repo root:
  python finetune_smiles.py \
      --data_path /data/chembl_33_filtered.csv \
      --pretrained louaaron/sedd-small \
      --output_dir ./runs/chembl_absorb \
      --epochs 30 \
      --batch_size 64 \
      --lr 3e-4 \
      --mode full

  # Lite mode (only embed + head), useful for small subsets:
  python finetune_smiles.py \
      --data_path /data/chembl_kinase.csv \
      --mode lite --epochs 100
"""

from __future__ import annotations

import argparse
import os
import math
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── SEDD imports (run from repo root) ────────────────────────────────────────
import graph_lib
import noise_lib
import losses as sedd_losses
from load_model import load_model

# ── Bio-SEDD imports ──────────────────────────────────────────────────────────
from smiles_tokenizer import SmilesTokenizer, VOCAB_SIZE, MASK_ID, PAD_ID
from smiles_dataset import ChEMBLDataset, collate_smiles


# ─────────────────────────────────────────────────────────────────────────────
# Model adaptation
# ─────────────────────────────────────────────────────────────────────────────

def swap_vocab_embeddings(
    model: nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    device: torch.device,
) -> nn.Module:
    """
    Replace embedding table and output projection for the SMILES vocabulary.
    Identical in structure to the protein version; factored separately so
    both scripts can be imported without conflicts.
    """
    d_model = None

    for attr in ["vocab_embed", "embedding", "tok_emb", "embed_tokens"]:
        if hasattr(model, attr):
            old_emb: nn.Embedding = getattr(model, attr)
            d_model = old_emb.embedding_dim
            new_emb = nn.Embedding(new_vocab_size, d_model, padding_idx=PAD_ID)
            nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
            setattr(model, attr, new_emb.to(device))
            print(f"  Swapped '{attr}': {old_vocab_size} → {new_vocab_size} tokens")
            break
    else:
        raise AttributeError(
            "Could not find embedding layer. "
            "Check attribute names in the SEDD model and update swap_vocab_embeddings()."
        )

    for attr in ["output_layer", "lm_head", "output_proj", "to_logits"]:
        if hasattr(model, attr):
            old_head: nn.Linear = getattr(model, attr)
            new_head = nn.Linear(d_model, new_vocab_size, bias=old_head.bias is not None)
            nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
            if new_head.bias is not None:
                nn.init.zeros_(new_head.bias)
            setattr(model, attr, new_head.to(device))
            print(f"  Swapped '{attr}': {old_vocab_size} → {new_vocab_size} outputs")
            break
    else:
        raise AttributeError(
            "Could not find output projection layer. "
            "Check attribute names and update swap_vocab_embeddings()."
        )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except embedding + output head (lite mode)."""
    total, frozen = 0, 0
    for name, param in model.named_parameters():
        total += 1
        is_head = any(k in name for k in [
            "vocab_embed", "embedding", "tok_emb", "embed_tokens",
            "output_layer", "lm_head", "output_proj", "to_logits",
        ])
        if not is_head:
            param.requires_grad = False
            frozen += 1
    print(f"  Lite mode: frozen {frozen}/{total} parameter groups.")


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    graph,
    noise,
    device: torch.device,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)

        loss = sedd_losses.get_loss_fn(graph, noise)(model, input_ids)
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    graph,
    noise,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        loss = sedd_losses.get_loss_fn(graph, noise)(model, input_ids)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SEDD on ChEMBL SMILES")

    # Data
    p.add_argument("--data_path", required=True,
                   help="CSV/TSV file or directory. Must contain a 'canonical_smiles' column.")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=128,
                   help="Max SMILES token length. Sequences exceeding this are DROPPED.")
    p.add_argument("--min_heavy_atoms", type=int, default=5)
    p.add_argument("--max_heavy_atoms", type=int, default=60)
    p.add_argument("--filter_valid", action="store_true", default=True,
                   help="Drop SMILES that RDKit cannot parse.")
    p.add_argument("--deduplicate", action="store_true", default=True)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (useful for ablations).")

    # Model
    p.add_argument("--pretrained", default="louaaron/sedd-small")
    p.add_argument("--graph_type", default="absorb", choices=["absorb", "uniform"])
    p.add_argument("--noise_type", default="loglinear", choices=["loglinear", "geometric"])
    p.add_argument("--mode", default="full", choices=["full", "lite"])

    # Training
    p.add_argument("--output_dir", default="./runs/chembl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="bio-sedd-chembl")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = SmilesTokenizer(
        max_length=args.max_length,
        canonicalize=True,   # canonical SMILES → consistent token sequences
    )
    print(f"SMILES vocabulary size: {tokenizer.vocab_size}")

    # ── Dataset ───────────────────────────────────────────────────────────
    full_ds = ChEMBLDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
        filter_valid=args.filter_valid,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        deduplicate=args.deduplicate,
    )

    val_size  = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {train_size:,}  |  Val: {val_size:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_smiles,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_smiles,
    )

    # ── Load pretrained SEDD ──────────────────────────────────────────────
    print(f"\nLoading pretrained SEDD from '{args.pretrained}' ...")
    model, graph, noise = load_model(args.pretrained, device=device)
    old_vocab_size = graph.dim

    # ── Override graph + noise for SMILES vocab ───────────────────────────
    if args.graph_type == "absorb":
        graph = graph_lib.Absorbing(VOCAB_SIZE)
    else:
        graph = graph_lib.Uniform(VOCAB_SIZE)

    if args.noise_type == "loglinear":
        noise = noise_lib.LogLinearNoise()
    else:
        noise = noise_lib.GeometricNoise()

    # ── Swap vocab ────────────────────────────────────────────────────────
    print("\nAdapting model vocabulary ...")
    model = swap_vocab_embeddings(model, old_vocab_size, VOCAB_SIZE, device)

    if args.mode == "lite":
        freeze_backbone(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = args.epochs * len(train_loader) // args.accum_steps

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"  Fine-tuning Bio-SEDD on ChEMBL  |  mode={args.mode}  |  vocab={VOCAB_SIZE}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            graph, noise, device,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
        )
        val_loss = evaluate(model, val_loader, graph, noise, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  t={elapsed:.1f}s"
        )

        if args.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "vocab_size": VOCAB_SIZE,
            }, os.path.join(args.output_dir, "best_checkpoint.pt"))
            print(f"  ✓ Best model saved (val={best_val_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "vocab_size": VOCAB_SIZE,
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pt"))

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best_checkpoint.pt')}")


if __name__ == "__main__":
    main()
