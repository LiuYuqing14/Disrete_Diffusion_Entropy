"""
evaluate_protein.py
-------------------
Zero-shot fitness evaluation on ProteinGym DMS benchmarks.

Method (pseudo-log-likelihood scoring)
---------------------------------------
For each mutant sequence x, we estimate the model's log-probability using
the SEDD masked-marginal scoring approach:

  score(x) = Σ_i  log p_θ(x_i | x_{-i}, t*)

where t* is a low noise level so that only a small fraction of tokens are
masked.  This is analogous to the masked marginal approach used in ESM-1v.

Alternatively, we use the SEDD score-entropy loss evaluated at t→0 as a
proxy for log p(x).

Metrics reported
----------------
  - Spearman ρ  (primary ProteinGym metric)
  - AUC (if DMS_score_bin is present)
  - NDCG@top10%

Usage
-----
  python evaluate_protein.py \
      --checkpoint ./protein_finetune/best_checkpoint.pt \
      --pretrained  louaaron/sedd-small \
      --dms_csv     /data/proteingym/substitutions/BLAT_ECOLX_Ranganathan2015.csv \
      --output_csv  results/BLAT_results.csv \
      --batch_size  32 \
      --score_method masked_marginal
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# ── SEDD imports ──────────────────────────────────────────────────────────────
import graph_lib
import noise_lib
from load_model import load_model

# ── Bio-SEDD imports ──────────────────────────────────────────────────────────
from protein_tokenizer import ProteinTokenizer, VOCAB_SIZE, MASK_ID, PAD_ID
from protein_dataset import ProteinGymDMSDataset, collate_dms


# ─────────────────────────────────────────────────────────────────────────────
# Scoring methods
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def masked_marginal_score(
    model: torch.nn.Module,
    input_ids: torch.Tensor,          # (B, L)
    mask_fraction: float = 0.15,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Masked-marginal log-likelihood estimate.

    For each sequence, randomly mask `mask_fraction` of positions
    `n_samples` times and average the cross-entropy log p(x_i | x_masked).

    Returns
    -------
    scores : torch.Tensor of shape (B,)
        Higher = more likely under the model.
    """
    model.eval()
    B, L = input_ids.shape
    input_ids = input_ids.to(device)
    all_log_probs = torch.zeros(B, device=device)

    for _ in range(n_samples):
        # ── Create masked input ─────────────────────────────────────────
        mask = torch.rand(B, L, device=device) < mask_fraction   # (B, L)
        # Never mask padding positions
        pad_mask = input_ids == PAD_ID
        mask = mask & ~pad_mask

        masked_input = input_ids.clone()
        masked_input[mask] = MASK_ID

        # ── Forward pass (model returns logits over vocab) ─────────────
        # SEDD models take (x, sigma) where sigma is the noise level.
        # At inference time, we use a very small sigma (≈ near-clean).
        sigma = torch.full((B,), 1e-3, device=device)
        logits = model(masked_input, sigma)                # (B, L, V)

        # ── Log-prob at masked positions ────────────────────────────────
        log_probs = F.log_softmax(logits, dim=-1)          # (B, L, V)
        # Gather the true token's log-prob
        true_log_probs = log_probs.gather(
            dim=-1,
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)                                       # (B, L)

        # Sum over masked positions only
        masked_log_probs = (true_log_probs * mask.float()).sum(dim=-1)  # (B,)
        n_masked = mask.float().sum(dim=-1).clamp(min=1.0)
        all_log_probs += masked_log_probs / n_masked

    return all_log_probs / n_samples


@torch.no_grad()
def wildtype_marginal_score(
    model: torch.nn.Module,
    input_ids: torch.Tensor,       # (B, L)  – mutant sequences
    wildtype_ids: torch.Tensor,    # (1, L)  – wildtype sequence
    mask_fraction: float = 0.15,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Wildtype-normalised score (mutant score − wildtype score).
    Used in ESM-1v / EVE-style evaluation to control for position bias.

    Returns
    -------
    delta_scores : torch.Tensor of shape (B,)
    """
    mut_scores = masked_marginal_score(
        model, input_ids, mask_fraction, n_samples, device
    )
    wt_expanded = wildtype_ids.expand(input_ids.shape[0], -1)
    wt_scores = masked_marginal_score(
        model, wt_expanded, mask_fraction, n_samples, device
    )
    return mut_scores - wt_scores


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_spearman(scores: np.ndarray, labels: np.ndarray) -> float:
    rho, pval = spearmanr(scores, labels)
    return float(rho)


def compute_auc(scores: np.ndarray, labels_bin: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels_bin, scores))
    except Exception:
        return float("nan")


def compute_ndcg_top_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k_fraction: float = 0.10,
) -> float:
    """NDCG for the top-k% predicted sequences."""
    try:
        from sklearn.metrics import ndcg_score
        k = max(1, int(len(scores) * k_fraction))
        return float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=k))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot ProteinGym evaluation")

    p.add_argument("--checkpoint", required=True,
                   help="Path to fine-tuned checkpoint (.pt) produced by finetune_protein.py")
    p.add_argument("--pretrained", default="louaaron/sedd-small",
                   help="HuggingFace repo for base SEDD architecture.")
    p.add_argument("--dms_csv", required=True,
                   help="ProteinGym assay CSV or directory of CSVs.")
    p.add_argument("--output_csv", default="./proteingym_results.csv")
    p.add_argument("--wildtype_seq", default=None,
                   help="Optional override for wildtype sequence.")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--score_method", default="masked_marginal",
                   choices=["masked_marginal", "wildtype_marginal"])
    p.add_argument("--mask_fraction", type=float, default=0.15)
    p.add_argument("--n_samples", type=int, default=10,
                   help="Monte-Carlo samples for masked-marginal scoring.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--graph_type", default="absorb")
    p.add_argument("--noise_type", default="loglinear")

    return p.parse_args()


def load_finetuned_model(args, device):
    """Load base SEDD architecture, then load fine-tuned weights."""
    print(f"Loading base architecture from '{args.pretrained}' ...")
    model, graph, noise = load_model(args.pretrained, device=device)

    # Rebuild graph / noise for protein vocab
    if args.graph_type == "absorb":
        graph = graph_lib.Absorbing(VOCAB_SIZE)
    else:
        graph = graph_lib.Uniform(VOCAB_SIZE)
    if args.noise_type == "loglinear":
        noise = noise_lib.LogLinearNoise()
    else:
        noise = noise_lib.GeometricNoise()

    # Swap vocab (must match what finetune_protein.py did)
    from finetune_protein import swap_vocab_embeddings
    old_vocab = graph_lib.Absorbing(50257).dim  # original GPT-2 size fallback
    # Try to read from saved config
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    old_vocab = saved_args.get("old_vocab_size", 50257)

    model = swap_vocab_embeddings(model, old_vocab, VOCAB_SIZE, device)

    print(f"Loading fine-tuned weights from '{args.checkpoint}' ...")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, graph, noise


def evaluate_single_assay(
    csv_path: str,
    model: torch.nn.Module,
    tokenizer: ProteinTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    ds = ProteinGymDMSDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        wildtype_seq=args.wildtype_seq,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_dms,
    )

    all_scores: List[float] = []
    all_fitness: List[float] = []

    for batch_seqs, batch_fitness in loader:
        input_ids = batch_seqs["input_ids"].to(device)

        if args.score_method == "wildtype_marginal" and args.wildtype_seq:
            wt_ids = tokenizer.batch_encode([args.wildtype_seq]).to(device)
            scores = wildtype_marginal_score(
                model, input_ids, wt_ids,
                mask_fraction=args.mask_fraction,
                n_samples=args.n_samples,
                device=device,
            )
        else:
            scores = masked_marginal_score(
                model, input_ids,
                mask_fraction=args.mask_fraction,
                n_samples=args.n_samples,
                device=device,
            )

        all_scores.extend(scores.cpu().numpy().tolist())
        all_fitness.extend(batch_fitness.numpy().tolist())

    scores_np  = np.array(all_scores)
    fitness_np = np.array(all_fitness)

    rho = compute_spearman(scores_np, fitness_np)
    ndcg = compute_ndcg_top_k(scores_np, fitness_np)

    result = {
        "assay": os.path.basename(csv_path),
        "n_variants": len(scores_np),
        "spearman_rho": round(rho, 4),
        "ndcg_top10": round(ndcg, 4),
    }

    # AUC if binary labels exist in original CSV
    df = pd.read_csv(csv_path)
    if "DMS_score_bin" in df.columns:
        labels_bin = df["DMS_score_bin"].values[: len(scores_np)]
        result["auc"] = round(compute_auc(scores_np, labels_bin), 4)

    return result, scores_np, fitness_np


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = ProteinTokenizer(max_length=args.max_length)
    model, graph, noise = load_finetuned_model(args, device)

    # ── Find assay CSVs ───────────────────────────────────────────────────
    import glob
    if os.path.isdir(args.dms_csv):
        csv_files = sorted(glob.glob(os.path.join(args.dms_csv, "*.csv")))
    else:
        csv_files = [args.dms_csv]

    print(f"\nEvaluating on {len(csv_files)} assay(s)...\n")

    all_results = []
    all_pred, all_true = [], []

    for csv_path in csv_files:
        result, scores, fitness = evaluate_single_assay(
            csv_path, model, tokenizer, args, device
        )
        all_results.append(result)
        all_pred.extend(scores.tolist())
        all_true.extend(fitness.tolist())

        print(
            f"  {result['assay']:<55}  "
            f"ρ={result['spearman_rho']:+.3f}  "
            f"NDCG={result['ndcg_top10']:.3f}"
            + (f"  AUC={result.get('auc', 'n/a')}" if 'auc' in result else "")
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    rhos = [r["spearman_rho"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"  Mean Spearman ρ : {np.mean(rhos):+.4f}")
    print(f"  Median ρ        : {np.median(rhos):+.4f}")
    print(f"  Overall ρ       : {compute_spearman(np.array(all_pred), np.array(all_true)):+.4f}")
    print(f"{'='*60}\n")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
