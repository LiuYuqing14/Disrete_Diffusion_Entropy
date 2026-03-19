"""
finetune_protein.py
-------------------
Fine-tune a pretrained SEDD model on ProteinGym protein sequences.

Key design decisions
--------------------
1. Vocabulary swap: SEDD was pretrained on text (vocab ~50k). We replace the
   token embedding and output projection with a protein vocabulary (25 tokens).
   The rest of the transformer weights are kept.

2. Graph / noise: We use `absorb` graph + `loglinear` noise — the best-performing
   SEDD configuration for language.  The absorbing token is <MASK> (id=1).

3. Objective: Standard SEDD score-entropy loss (losses.py from the original repo).

4. Two training modes:
   (a) full fine-tune  – all weights updated (good if you have ≥10k sequences)
   (b) lite fine-tune  – only embedding + output head updated (few-shot regime)

Usage
-----
  # From inside the Score-Entropy-Discrete-Diffusion repo root:
  python finetune_protein.py \
      --data_path /data/proteingym/substitutions/ \
      --pretrained louaaron/sedd-small \
      --output_dir ./protein_finetune \
      --epochs 20 \
      --batch_size 16 \
      --lr 3e-4 \
      --mode full

Requirements
------------
  pip install pandas scipy tqdm wandb
  # Clone SEDD repo and run from its root so that graph_lib, noise_lib,
  # losses, load_model, etc. are importable.
"""

from __future__ import annotations

import argparse
import os
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── SEDD imports (must run from the repo root) ────────────────────────────────
import graph_lib
import noise_lib
import losses as sedd_losses
from load_model import load_model

# ── Bio-SEDD imports ──────────────────────────────────────────────────────────
from protein_tokenizer import ProteinTokenizer, VOCAB_SIZE, MASK_ID, PAD_ID
from protein_dataset import ProteinGymSequenceDataset, collate_sequences


# ─────────────────────────────────────────────────────────────────────────────
# Model adaptation helpers
# ─────────────────────────────────────────────────────────────────────────────

def swap_vocab_embeddings(
    model: nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    device: torch.device,
) -> nn.Module:
    """
    Replace the token embedding table and the output projection (lm_head /
    output layer) with randomly initialised layers sized for `new_vocab_size`.

    SEDD's DiT-style model stores:
      model.vocab_embed   – nn.Embedding(V_old, d_model)
      model.output_layer  – nn.Linear(d_model, V_old, bias=False)
    (attribute names may differ; we probe both common patterns)
    """
    d_model = None

    # ── Embedding ────────────────────────────────────────────────────────────
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
            "Could not find the embedding layer. "
            "Check the model attribute name and update `swap_vocab_embeddings`."
        )

    # ── Output head ──────────────────────────────────────────────────────────
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
            "Could not find the output projection layer. "
            "Check the model attribute name and update `swap_vocab_embeddings`."
        )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except embedding and output head."""
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
    print(f"  Lite mode: frozen {frozen}/{total} parameter groups "
          f"(only embeddings + head are trainable).")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
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
        input_ids = batch["input_ids"].to(device)       # (B, L)
        # attention_mask not used by SEDD loss directly, but available

        # ── SEDD score-entropy loss ───────────────────────────────────────
        # losses.get_loss_fn returns a function: loss_fn(model, batch) -> scalar
        # We call the internal computation directly for flexibility.
        loss = sedd_losses.get_loss_fn(graph, noise)(model, input_ids)
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


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
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SEDD on ProteinGym")

    # Data
    p.add_argument("--data_path", required=True,
                   help="Path to ProteinGym CSV file or directory of CSVs.")
    p.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction of data held out for validation.")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max sequence length (truncate longer sequences).")
    p.add_argument("--min_fitness", type=float, default=None,
                   help="Only train on sequences with DMS_score >= this value.")

    # Model
    p.add_argument("--pretrained", default="louaaron/sedd-small",
                   help="HuggingFace repo id or local path for pretrained SEDD.")
    p.add_argument("--graph_type", default="absorb", choices=["absorb", "uniform"])
    p.add_argument("--noise_type", default="loglinear", choices=["loglinear", "geometric"])
    p.add_argument("--mode", default="full", choices=["full", "lite"],
                   help="'full' = fine-tune all weights; 'lite' = only head+embedding.")

    # Training
    p.add_argument("--output_dir", default="./protein_finetune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs.")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="bio-sedd-protein")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB (optional) ──────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = ProteinTokenizer(
        add_bos=False,
        add_eos=False,
        max_length=args.max_length,
    )
    print(f"Protein vocabulary size: {tokenizer.vocab_size}")

    # ── Dataset ───────────────────────────────────────────────────────────
    full_ds = ProteinGymSequenceDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        include_wildtype=True,
        min_fitness=args.min_fitness,
    )

    val_size = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {train_size:,}  |  Val: {val_size:,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )

    # ── Load pretrained SEDD ──────────────────────────────────────────────
    print(f"\nLoading pretrained SEDD from '{args.pretrained}' ...")
    model, graph, noise = load_model(args.pretrained, device=device)
    old_vocab_size = graph.dim   # original text vocabulary size

    print(f"  Original vocab size  : {old_vocab_size}")
    print(f"  Protein vocab size   : {VOCAB_SIZE}")

    # ── Override graph & noise for protein domain ─────────────────────────
    # absorb graph: absorbing token = MASK_ID (1) in our protein vocab
    if args.graph_type == "absorb":
        graph = graph_lib.Absorbing(VOCAB_SIZE)
    else:
        graph = graph_lib.Uniform(VOCAB_SIZE)

    if args.noise_type == "loglinear":
        noise = noise_lib.LogLinearNoise()
    else:
        noise = noise_lib.GeometricNoise()

    # ── Swap embedding / head for protein vocab ───────────────────────────
    print("\nAdapting model vocabulary ...")
    model = swap_vocab_embeddings(model, old_vocab_size, VOCAB_SIZE, device)

    if args.mode == "lite":
        freeze_backbone(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = args.epochs * len(train_loader) // args.accum_steps

    def lr_lambda(current_step: int) -> float:
        """Linear warmup + cosine decay."""
        if current_step < args.warmup_steps:
            return current_step / max(1, args.warmup_steps)
        progress = (current_step - args.warmup_steps) / max(
            1, total_steps - args.warmup_steps
        )
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"  Starting fine-tuning  |  mode={args.mode}  |  epochs={args.epochs}")
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
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={lr_now:.2e}  t={elapsed:.1f}s"
        )

        if args.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "lr": lr_now, "epoch": epoch})

        # ── Checkpointing ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)

    print(f"\nFinished. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved at: {os.path.join(args.output_dir, 'best_checkpoint.pt')}")


if __name__ == "__main__":
    main()
