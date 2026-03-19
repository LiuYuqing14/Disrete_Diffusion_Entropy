"""
protein_dataset.py
------------------
PyTorch Dataset classes for ProteinGym fine-tuning.

ProteinGym structure (substitution benchmark):
  Each assay CSV has columns:
    mutant          – mutation string, e.g. "A2G:L5R"
    mutated_sequence – full sequence with mutations applied
    DMS_score        – experimental fitness (float)
    DMS_score_bin    – binarised fitness (0 / 1)

Reference:
  Notin et al., ProteinGym: Large-scale benchmarks for protein fitness
  prediction and design (NeurIPS 2023).

Usage
-----
  # Fine-tuning dataset (sequence-level, no fitness labels needed)
  from protein_dataset import ProteinGymSequenceDataset
  ds = ProteinGymSequenceDataset("proteingym/substitutions/", tokenizer, max_length=512)

  # Evaluation dataset (returns sequence + fitness label for zero-shot scoring)
  from protein_dataset import ProteinGymDMSDataset
  ds = ProteinGymDMSDataset("proteingym/substitutions/BLAT_ECOLX_Ranganathan2015.csv", tokenizer)
"""

from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from protein_tokenizer import ProteinTokenizer, PAD_ID


# ── Helper ────────────────────────────────────────────────────────────────────

def _apply_mutations(wildtype: str, mutant_str: str) -> str:
    """
    Apply a mutation string like 'A2G:L5R' to a wildtype sequence.
    Falls back to returning the wildtype if anything looks wrong.
    """
    seq = list(wildtype)
    for mut in mutant_str.split(":"):
        if not mut:
            continue
        try:
            wt_aa = mut[0]
            pos = int(mut[1:-1]) - 1   # 1-indexed → 0-indexed
            mt_aa = mut[-1]
            if 0 <= pos < len(seq):
                seq[pos] = mt_aa
        except (ValueError, IndexError):
            pass
    return "".join(seq)


# ── Dataset 1 : fine-tuning (sequence-only) ───────────────────────────────────

class ProteinGymSequenceDataset(Dataset):
    """
    Iterates over *all* mutated sequences across one or more ProteinGym assay
    CSVs.  Returns only token ids (no fitness labels) — suitable for the SEDD
    cross-entropy / score-entropy training objective.

    Parameters
    ----------
    data_path : str
        Path to a single CSV file **or** a directory of CSVs.
    tokenizer : ProteinTokenizer
    max_length : int
        Sequences longer than this are truncated.
    include_wildtype : bool
        If True, also include the wildtype sequence once per assay.
    min_fitness : float | None
        If set, only keep sequences with DMS_score >= min_fitness.
        Useful to bias training toward high-fitness variants.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ProteinTokenizer,
        max_length: int = 512,
        include_wildtype: bool = True,
        min_fitness: Optional[float] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences: List[str] = []

        csv_files = self._find_csvs(data_path)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found at: {data_path}")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            self._load_assay(df, include_wildtype, min_fitness)

        print(f"[ProteinGymSequenceDataset] Loaded {len(self.sequences):,} sequences "
              f"from {len(csv_files)} assay(s).")

    @staticmethod
    def _find_csvs(path: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        return sorted(glob.glob(os.path.join(path, "*.csv")))

    def _load_assay(
        self,
        df: pd.DataFrame,
        include_wildtype: bool,
        min_fitness: Optional[float],
    ) -> None:
        # ── filter by fitness ──────────────────────────────────────────────
        if min_fitness is not None and "DMS_score" in df.columns:
            df = df[df["DMS_score"] >= min_fitness]

        # ── collect sequences ──────────────────────────────────────────────
        if "mutated_sequence" in df.columns:
            seqs = df["mutated_sequence"].dropna().tolist()
            self.sequences.extend(seqs)

        elif "mutant" in df.columns and "sequence" in df.columns:
            # Some ProteinGym files have a 'sequence' column for the wildtype
            wt = df["sequence"].iloc[0]
            for mut in df["mutant"].dropna():
                self.sequences.append(_apply_mutations(wt, mut))
            if include_wildtype:
                self.sequences.append(wt)
        else:
            print(f"  [WARN] Unexpected columns: {list(df.columns)[:6]} – skipping.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx][: self.max_length]
        ids = self.tokenizer.encode(seq)
        return torch.tensor(ids, dtype=torch.long)


# ── Dataset 2 : evaluation (sequence + fitness score) ────────────────────────

class ProteinGymDMSDataset(Dataset):
    """
    Single-assay dataset for zero-shot fitness evaluation.

    Returns (token_ids, fitness_score) pairs.
    Used in evaluate_protein.py to compute Spearman ρ between
    model log-likelihood and experimental DMS scores.

    Parameters
    ----------
    csv_path : str
        Path to a single ProteinGym assay CSV.
    tokenizer : ProteinTokenizer
    max_length : int
    wildtype_seq : str | None
        Override the wildtype sequence.  If None, tries to infer from
        'mutated_sequence' or a 'sequence' column.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: ProteinTokenizer,
        max_length: int = 512,
        wildtype_seq: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path)
        self._validate(df)

        self.sequences: List[str] = []
        self.fitness: List[float] = []
        self.mutants: List[str] = []

        for _, row in df.iterrows():
            seq = self._get_sequence(row, wildtype_seq)
            if seq is None:
                continue
            self.sequences.append(seq[: max_length])
            self.fitness.append(float(row.get("DMS_score", 0.0)))
            self.mutants.append(str(row.get("mutant", "")))

        print(f"[ProteinGymDMSDataset] {os.path.basename(csv_path)} — "
              f"{len(self.sequences):,} variants loaded.")

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if "DMS_score" not in df.columns:
            raise ValueError("CSV must contain a 'DMS_score' column.")

    @staticmethod
    def _get_sequence(
        row: pd.Series,
        wildtype_seq: Optional[str],
    ) -> Optional[str]:
        if "mutated_sequence" in row.index and pd.notna(row["mutated_sequence"]):
            return str(row["mutated_sequence"])
        if wildtype_seq is not None and "mutant" in row.index:
            return _apply_mutations(wildtype_seq, str(row["mutant"]))
        return None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        ids = self.tokenizer.encode(self.sequences[idx])
        return torch.tensor(ids, dtype=torch.long), self.fitness[idx]

    @property
    def fitness_array(self) -> List[float]:
        return self.fitness


# ── Collate function ──────────────────────────────────────────────────────────

def collate_sequences(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Pad a list of 1-D token tensors to the same length.
    Returns a dict with keys 'input_ids' and 'attention_mask'.
    """
    max_len = max(t.shape[0] for t in batch)
    padded, masks = [], []
    for t in batch:
        pad_len = max_len - t.shape[0]
        padded.append(torch.cat([t, torch.full((pad_len,), PAD_ID, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(t.shape[0]), torch.zeros(pad_len)]).long())
    return {
        "input_ids": torch.stack(padded),
        "attention_mask": torch.stack(masks),
    }


def collate_dms(
    batch: List[Tuple[torch.Tensor, float]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collate for ProteinGymDMSDataset."""
    seqs, scores = zip(*batch)
    return collate_sequences(list(seqs)), torch.tensor(scores, dtype=torch.float)
