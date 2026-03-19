"""
smiles_dataset.py
-----------------
PyTorch Dataset classes for ChEMBL fine-tuning of Bio-SEDD.

ChEMBL data sources
-------------------
ChEMBL can be downloaded in several formats.  We support all common ones:

  1. ChEMBL SQLite dump  →  query via chembl_webresource_client or sqlite3
  2. Filtered CSV / TSV  →  e.g. chembl_smiles.csv with column 'canonical_smiles'
  3. SDF files           →  parsed via RDKit (optional)
  4. ZINC250k / GuacaMol benchmark CSVs (same column name convention)

The primary column we look for is 'canonical_smiles'.
Fallbacks: 'smiles', 'SMILES', 'Smiles'.

Usage
-----
  from smiles_dataset import ChEMBLDataset, collate_smiles
  ds = ChEMBLDataset("/data/chembl_33.csv", tokenizer, max_length=128)
  loader = DataLoader(ds, batch_size=64, collate_fn=collate_smiles)
"""

from __future__ import annotations

import os
import glob
from typing import List, Optional, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from smiles_tokenizer import SmilesTokenizer, PAD_ID


# ── Validity filter (optional, requires RDKit) ────────────────────────────────

def _is_valid(smi: str) -> bool:
    """Return True if RDKit can parse the SMILES. Always True if RDKit absent."""
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smi) is not None
    except ImportError:
        return True


# ── Column name detection ─────────────────────────────────────────────────────

_SMILES_COLS = ["canonical_smiles", "smiles", "SMILES", "Smiles", "structure"]

def _find_smiles_col(df: pd.DataFrame) -> str:
    for col in _SMILES_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        f"No SMILES column found. Looked for: {_SMILES_COLS}. "
        f"Got: {list(df.columns)[:8]}"
    )


# ── Dataset 1: fine-tuning (sequence only) ────────────────────────────────────

class ChEMBLDataset(Dataset):
    """
    Loads SMILES strings from ChEMBL CSV / TSV files for SEDD fine-tuning.
    Returns only token id tensors — no property labels needed.

    Parameters
    ----------
    data_path    : path to CSV/TSV file or directory of such files
    tokenizer    : SmilesTokenizer instance
    max_length   : sequences longer than this are dropped (not truncated),
                   to avoid partial SMILES that are chemically meaningless
    max_samples  : cap total dataset size (useful for ablations)
    filter_valid : if True and RDKit available, drop invalid SMILES
    min_heavy_atoms : drop molecules with fewer than this many heavy atoms
    max_heavy_atoms : drop molecules with more  than this many heavy atoms
    deduplicate  : remove exact-duplicate SMILES strings
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: SmilesTokenizer,
        max_length: int = 128,
        max_samples: Optional[int] = None,
        filter_valid: bool = True,
        min_heavy_atoms: int = 5,
        max_heavy_atoms: int = 60,
        deduplicate: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles: List[str] = []

        csv_files = self._find_files(data_path)
        if not csv_files:
            raise FileNotFoundError(f"No CSV/TSV files found at: {data_path}")

        raw: List[str] = []
        for path in csv_files:
            raw.extend(self._load_file(path))

        print(f"[ChEMBLDataset] Raw SMILES loaded: {len(raw):,} from {len(csv_files)} file(s)")

        if deduplicate:
            raw = list(dict.fromkeys(raw))
            print(f"  After dedup: {len(raw):,}")

        # Filter
        kept = 0
        for smi in raw:
            if max_samples and kept >= max_samples:
                break
            smi = smi.strip()
            if not smi:
                continue
            # Length filter (drop, don't truncate — partial SMILES are invalid)
            encoded = tokenizer.encode(smi)
            if len(encoded) > max_length or len(encoded) < 3:
                continue
            # Validity filter
            if filter_valid and not _is_valid(smi):
                continue
            # Heavy atom count filter
            if not self._check_heavy_atoms(smi, min_heavy_atoms, max_heavy_atoms):
                continue
            self.smiles.append(smi)
            kept += 1

        print(f"  After filtering: {len(self.smiles):,} sequences kept")

    @staticmethod
    def _find_files(path: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        files += sorted(glob.glob(os.path.join(path, "*.tsv")))
        return files

    @staticmethod
    def _load_file(path: str) -> List[str]:
        sep = "\t" if path.endswith(".tsv") else ","
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            col = _find_smiles_col(df)
            return df[col].dropna().astype(str).tolist()
        except Exception as e:
            print(f"  [WARN] Could not load {os.path.basename(path)}: {e}")
            return []

    @staticmethod
    def _check_heavy_atoms(smi: str, min_ha: int, max_ha: int) -> bool:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return False
            n = mol.GetNumHeavyAtoms()
            return min_ha <= n <= max_ha
        except ImportError:
            return True  # skip filter if no RDKit

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.smiles[idx])
        return torch.tensor(ids, dtype=torch.long)


# ── Dataset 2: property-labelled (for conditional gen / evaluation) ───────────

class ChEMBLPropertyDataset(Dataset):
    """
    Dataset with SMILES + molecular property labels.
    Used for evaluating zero-shot property prediction (e.g. logP, QED).

    Parameters
    ----------
    data_path        : CSV file with SMILES + property columns
    tokenizer        : SmilesTokenizer
    property_cols    : list of column names to use as labels
                       (e.g. ['standard_value', 'pchembl_value', 'qed'])
    max_length       : max token length
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: SmilesTokenizer,
        property_cols: Optional[List[str]] = None,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(data_path)
        smiles_col = _find_smiles_col(df)

        if property_cols is None:
            # Auto-detect numeric columns
            property_cols = [
                c for c in df.columns
                if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])
            ][:4]  # cap at 4 properties

        self.property_cols = property_cols
        self.smiles: List[str] = []
        self.properties: List[List[float]] = []

        for _, row in df.iterrows():
            smi = str(row[smiles_col]).strip()
            if not smi or not _is_valid(smi):
                continue
            ids = tokenizer.encode(smi)
            if len(ids) > max_length or len(ids) < 3:
                continue
            props = [float(row.get(c, float("nan"))) for c in property_cols]
            self.smiles.append(smi)
            self.properties.append(props)

        print(
            f"[ChEMBLPropertyDataset] {len(self.smiles):,} molecules, "
            f"properties: {property_cols}"
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.encode(self.smiles[idx])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.properties[idx], dtype=torch.float),
        )


# ── Collate functions ─────────────────────────────────────────────────────────

def collate_smiles(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pad a batch of 1-D token tensors. Returns input_ids + attention_mask."""
    max_len = max(t.shape[0] for t in batch)
    padded, masks = [], []
    for t in batch:
        pad_len = max_len - t.shape[0]
        padded.append(
            torch.cat([t, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
        )
        masks.append(
            torch.cat([torch.ones(t.shape[0]), torch.zeros(pad_len)]).long()
        )
    return {
        "input_ids": torch.stack(padded),
        "attention_mask": torch.stack(masks),
    }


def collate_smiles_with_props(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    seqs, props = zip(*batch)
    return collate_smiles(list(seqs)), torch.stack(props)
