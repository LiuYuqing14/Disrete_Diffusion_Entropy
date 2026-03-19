"""
evaluate_smiles.py
------------------
Molecular generation + evaluation for Bio-SEDD fine-tuned on ChEMBL.

Unlike ProteinGym (which evaluates fitness correlation), molecule evaluation
focuses on generation quality. We sample from the model and report:

  Metrics
  -------
  validity     – fraction of generated SMILES parseable by RDKit
  uniqueness   – fraction of valid molecules that are unique
  novelty      – fraction of unique molecules NOT in the training set
  diversity    – average pairwise Tanimoto distance (Morgan fingerprints)
  QED          – mean drug-likeness score (Quantitative Estimate of Druglikeness)
  SA score     – mean synthetic accessibility score (1=easy, 10=hard)
  logP         – mean Wildman-Crippen logP
  Frac_valid_unique – validity × uniqueness (primary benchmark metric)

These metrics match the GuacaMol / MOSES benchmarks, allowing comparison
with REINVENT, JT-VAE, GraphAF, etc.

Scoring mode (zero-shot fitness proxy, optional)
-------------------------------------------------
Similar to ProteinGym, we also support scoring a set of known molecules
using masked marginal log-likelihood, then correlating with a property label
(e.g. pChEMBL activity).

Usage
-----
  # Generate and evaluate 10,000 molecules:
  python evaluate_smiles.py \
      --checkpoint ./runs/chembl/best_checkpoint.pt \
      --pretrained  louaaron/sedd-small \
      --mode        generate \
      --n_samples   10000 \
      --steps       1000 \
      --output_csv  results/generated.csv

  # Score a CSV of known molecules against a property:
  python evaluate_smiles.py \
      --checkpoint  ./runs/chembl/best_checkpoint.pt \
      --pretrained  louaaron/sedd-small \
      --mode        score \
      --score_csv   /data/chembl_activity.csv \
      --score_prop  pchembl_value \
      --output_csv  results/scored.csv
"""

from __future__ import annotations

import argparse
import os
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# ── SEDD imports ──────────────────────────────────────────────────────────────
import graph_lib
import noise_lib
import sampling as sedd_sampling
from load_model import load_model

# ── Bio-SEDD imports ──────────────────────────────────────────────────────────
from smiles_tokenizer import SmilesTokenizer, VOCAB_SIZE, MASK_ID, PAD_ID
from smiles_dataset import ChEMBLDataset, ChEMBLPropertyDataset, collate_smiles


# ─────────────────────────────────────────────────────────────────────────────
# RDKit helpers — all optional-guarded
# ─────────────────────────────────────────────────────────────────────────────

def _require_rdkit():
    try:
        import rdkit  # noqa: F401
    except ImportError:
        raise ImportError(
            "RDKit is required for molecule evaluation. "
            "Install with: conda install -c conda-forge rdkit"
        )


def compute_validity(smiles_list: List[str]) -> Tuple[List[str], float]:
    """Return (valid_smiles, validity_rate)."""
    _require_rdkit()
    from rdkit import Chem
    valid = []
    for smi in smiles_list:
        smi = smi.strip()
        if smi and Chem.MolFromSmiles(smi) is not None:
            valid.append(smi)
    return valid, len(valid) / max(len(smiles_list), 1)


def compute_uniqueness(valid_smiles: List[str]) -> Tuple[List[str], float]:
    """Return (unique_smiles, uniqueness_rate)."""
    _require_rdkit()
    from rdkit import Chem
    canonical = []
    seen = set()
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        csmi = Chem.MolToSmiles(mol)
        if csmi not in seen:
            canonical.append(csmi)
            seen.add(csmi)
    return canonical, len(canonical) / max(len(valid_smiles), 1)


def compute_novelty(unique_smiles: List[str], train_smiles: List[str]) -> float:
    """Fraction of generated molecules NOT in the training set."""
    train_set = set(train_smiles)
    novel = sum(1 for s in unique_smiles if s not in train_set)
    return novel / max(len(unique_smiles), 1)


def compute_diversity(unique_smiles: List[str], n_sample: int = 1000) -> float:
    """
    Mean pairwise Tanimoto distance on Morgan fingerprints.
    Sampled over min(n_sample, len) molecules for efficiency.
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mols = [Chem.MolFromSmiles(s) for s in unique_smiles if Chem.MolFromSmiles(s)]
    if len(mols) < 2:
        return 0.0

    sample = mols[:n_sample]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in sample]
    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, min(i + 50, len(fps))):  # compare to 50 nearest
            distances.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(distances)) if distances else 0.0


def compute_qed(smiles_list: List[str]) -> float:
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import QED
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scores.append(QED.qed(mol))
            except Exception:
                pass
    return float(np.mean(scores)) if scores else 0.0


def compute_sa_score(smiles_list: List[str]) -> float:
    """Synthetic Accessibility score via sascorer (RDKit contrib)."""
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        from rdkit import Chem
        scores = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scores.append(sascorer.calculateScore(mol))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return float("nan")


def compute_logp(smiles_list: List[str]) -> float:
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scores.append(Descriptors.MolLogP(mol))
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_smiles(
    model: torch.nn.Module,
    tokenizer: SmilesTokenizer,
    graph,
    noise,
    n_samples: int,
    seq_length: int,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> List[str]:
    """
    Generate molecules using SEDD's reverse diffusion sampler.

    Starts from a fully-masked sequence (all tokens = MASK_ID) and
    iteratively denoises using the learned score network.
    """
    model.eval()
    all_smiles: List[str] = []
    n_batches = math.ceil(n_samples / batch_size)

    for b in range(n_batches):
        current_batch = min(batch_size, n_samples - b * batch_size)

        # Start from noise (all-mask for absorb graph, random for uniform)
        if hasattr(graph, 'absorb'):  # Absorbing graph
            x = torch.full(
                (current_batch, seq_length), MASK_ID,
                dtype=torch.long, device=device
            )
        else:  # Uniform graph
            x = torch.randint(
                0, VOCAB_SIZE, (current_batch, seq_length),
                device=device
            )

        # SEDD reverse sampling
        # sampling.get_pc_sampler returns a sampler function
        sampler = sedd_sampling.get_pc_sampler(
            graph, noise, (current_batch, seq_length),
            "analytic",       # predictor
            "none",           # corrector
            device=device,
        )
        x_gen = sampler(model)  # (B, L)

        # Decode to SMILES strings
        for row in x_gen:
            smi = tokenizer.decode(row, skip_special=True)
            all_smiles.append(smi)

        print(f"  Generated batch {b+1}/{n_batches} ({len(all_smiles)} total)", end="\r")

    print()
    return all_smiles


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot scoring (masked marginal, same approach as proteins)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def masked_marginal_score(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask_fraction: float = 0.15,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Estimate log p(x) via masked marginal averaging.
    Same method as evaluate_protein.py — architecture-agnostic.
    """
    model.eval()
    B, L = input_ids.shape
    input_ids = input_ids.to(device)
    all_log_probs = torch.zeros(B, device=device)

    for _ in range(n_samples):
        mask = torch.rand(B, L, device=device) < mask_fraction
        pad_mask = input_ids == PAD_ID
        mask = mask & ~pad_mask

        masked_input = input_ids.clone()
        masked_input[mask] = MASK_ID

        sigma = torch.full((B,), 1e-3, device=device)
        logits = model(masked_input, sigma)          # (B, L, V)
        log_probs = F.log_softmax(logits, dim=-1)
        true_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        masked_sum = (true_log_probs * mask.float()).sum(-1)
        n_masked = mask.float().sum(-1).clamp(min=1.0)
        all_log_probs += masked_sum / n_masked

    return all_log_probs / n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned_model(args, device):
    print(f"Loading base architecture from '{args.pretrained}' ...")
    model, graph, noise = load_model(args.pretrained, device=device)

    if args.graph_type == "absorb":
        graph = graph_lib.Absorbing(VOCAB_SIZE)
    else:
        graph = graph_lib.Uniform(VOCAB_SIZE)

    if args.noise_type == "loglinear":
        noise = noise_lib.LogLinearNoise()
    else:
        noise = noise_lib.GeometricNoise()

    from finetune_smiles import swap_vocab_embeddings
    ckpt = torch.load(args.checkpoint, map_location=device)
    old_vocab = ckpt.get("args", {}).get("old_vocab_size", 50257)

    model = swap_vocab_embeddings(model, old_vocab, VOCAB_SIZE, device)
    print(f"Loading weights from '{args.checkpoint}' ...")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, graph, noise


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Bio-SEDD on ChEMBL molecules")

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pretrained", default="louaaron/sedd-small")
    p.add_argument("--mode", default="generate",
                   choices=["generate", "score"],
                   help="'generate' = sample new molecules and compute metrics; "
                        "'score' = score existing molecules, correlate with property")
    p.add_argument("--output_csv", default="./chembl_results.csv")

    # Generation args
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--seq_length", type=int, default=80,
                   help="Length of sequences to generate (in tokens).")
    p.add_argument("--steps", type=int, default=1000,
                   help="Number of reverse diffusion steps.")
    p.add_argument("--gen_batch_size", type=int, default=256)
    p.add_argument("--train_smiles_csv", default=None,
                   help="CSV of training molecules for novelty calculation.")

    # Scoring args
    p.add_argument("--score_csv", default=None,
                   help="CSV with SMILES + property column for scoring mode.")
    p.add_argument("--score_prop", default=None,
                   help="Column name for the property to correlate with.")
    p.add_argument("--mask_fraction", type=float, default=0.15)
    p.add_argument("--n_mask_samples", type=int, default=10)
    p.add_argument("--score_batch_size", type=int, default=64)

    p.add_argument("--graph_type", default="absorb")
    p.add_argument("--noise_type", default="loglinear")
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def run_generation(args, model, tokenizer, graph, noise, device):
    print(f"\nGenerating {args.n_samples:,} molecules ({args.steps} diffusion steps)...")
    raw = generate_smiles(
        model, tokenizer, graph, noise,
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        steps=args.steps,
        batch_size=args.gen_batch_size,
        device=device,
    )

    print("\nComputing metrics ...")
    valid, validity   = compute_validity(raw)
    unique, uniqueness = compute_uniqueness(valid)

    novelty = float("nan")
    if args.train_smiles_csv:
        train_df = pd.read_csv(args.train_smiles_csv)
        col = next((c for c in ["canonical_smiles","smiles","SMILES"] if c in train_df.columns), None)
        if col:
            train_smiles = train_df[col].dropna().tolist()
            novelty = compute_novelty(unique, train_smiles)

    diversity = compute_diversity(unique)
    qed       = compute_qed(unique)
    sa        = compute_sa_score(unique)
    logp      = compute_logp(unique)

    results = {
        "n_generated":   len(raw),
        "validity":      round(validity, 4),
        "uniqueness":    round(uniqueness, 4),
        "novelty":       round(novelty, 4) if not math.isnan(novelty) else "n/a",
        "diversity":     round(diversity, 4),
        "frac_valid_unique": round(validity * uniqueness, 4),
        "mean_QED":      round(qed, 4),
        "mean_SA":       round(sa, 4) if not math.isnan(sa) else "n/a",
        "mean_logP":     round(logp, 4),
    }

    print(f"\n{'='*50}")
    for k, v in results.items():
        print(f"  {k:<24} {v}")
    print(f"{'='*50}\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_out = pd.DataFrame({"smiles": unique})
    df_out.to_csv(args.output_csv, index=False)
    pd.DataFrame([results]).to_csv(
        args.output_csv.replace(".csv", "_metrics.csv"), index=False
    )
    print(f"Generated molecules saved to: {args.output_csv}")
    print(f"Metrics saved to:            {args.output_csv.replace('.csv', '_metrics.csv')}")


def run_scoring(args, model, tokenizer, device):
    from scipy.stats import spearmanr

    if args.score_csv is None or args.score_prop is None:
        raise ValueError("--score_csv and --score_prop required for score mode")

    ds = ChEMBLPropertyDataset(
        args.score_csv, tokenizer,
        property_cols=[args.score_prop],
        max_length=128,
    )
    from smiles_dataset import collate_smiles_with_props
    loader = DataLoader(
        ds, batch_size=args.score_batch_size,
        collate_fn=collate_smiles_with_props,
        num_workers=args.num_workers,
    )

    all_scores, all_props = [], []
    print(f"\nScoring {len(ds):,} molecules ...")

    for batch_seqs, batch_props in loader:
        input_ids = batch_seqs["input_ids"].to(device)
        scores = masked_marginal_score(
            model, input_ids,
            mask_fraction=args.mask_fraction,
            n_samples=args.n_mask_samples,
            device=device,
        )
        all_scores.extend(scores.cpu().tolist())
        all_props.extend(batch_props[:, 0].tolist())

    scores_np = np.array(all_scores)
    props_np  = np.array(all_props)
    valid_mask = ~np.isnan(props_np)
    rho, pval = spearmanr(scores_np[valid_mask], props_np[valid_mask])

    print(f"\nSpearman ρ ({args.score_prop}): {rho:+.4f}  (p={pval:.3e})")

    df_out = pd.DataFrame({
        "smiles": ds.smiles,
        "model_score": scores_np,
        args.score_prop: props_np,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Scores saved to: {args.output_csv}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SmilesTokenizer(max_length=128, canonicalize=True)
    model, graph, noise = load_finetuned_model(args, device)

    if args.mode == "generate":
        run_generation(args, model, tokenizer, graph, noise, device)
    else:
        run_scoring(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
