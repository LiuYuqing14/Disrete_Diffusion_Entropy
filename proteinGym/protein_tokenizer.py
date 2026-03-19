"""
protein_tokenizer.py
--------------------
Amino-acid tokenizer for Bio-SEDD.

Vocabulary (size = 25):
  0  : <PAD>   – padding token
  1  : <MASK>  – absorbing-state / mask token used by SEDD graph
  2  : <BOS>   – begin-of-sequence (optional, useful for conditional gen.)
  3  : <EOS>   – end-of-sequence
  4-23: the 20 canonical amino acids (alphabetically ordered)
  24 : <UNK>   – rare/non-standard residues

Design note:
  SEDD's absorb graph uses a single "absorbing" token id.
  We map that to token id 1 (<MASK>), matching the convention in
  the original SEDD codebase (graph_lib.py uses vocab[0] for uniform,
  vocab[-1] for absorb by default — we override via config).
"""

from __future__ import annotations
from typing import List, Union
import torch

# ── Canonical 20 amino acids (sorted) ────────────────────────────────────────
_AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
assert len(_AMINO_ACIDS) == 20

# ── Special tokens ────────────────────────────────────────────────────────────
_SPECIAL = ["<PAD>", "<MASK>", "<BOS>", "<EOS>", "<UNK>"]

# ── Build vocab ───────────────────────────────────────────────────────────────
_VOCAB: List[str] = _SPECIAL + _AMINO_ACIDS   # length = 25

_TOKEN2ID = {tok: idx for idx, tok in enumerate(_VOCAB)}
_ID2TOKEN = {idx: tok for tok, idx in _TOKEN2ID.items()}

# Convenient constants (used across the codebase)
PAD_ID   = _TOKEN2ID["<PAD>"]    # 0
MASK_ID  = _TOKEN2ID["<MASK>"]   # 1  ← absorbing token
BOS_ID   = _TOKEN2ID["<BOS>"]    # 2
EOS_ID   = _TOKEN2ID["<EOS>"]    # 3
UNK_ID   = _TOKEN2ID["<UNK>"]    # 4  (index 4, but let's keep < UNK > at 4)
VOCAB_SIZE = len(_VOCAB)         # 25


class ProteinTokenizer:
    """
    Character-level tokenizer for protein sequences.

    Parameters
    ----------
    add_bos : bool
        Prepend <BOS> token.
    add_eos : bool
        Append <EOS> token.
    max_length : int | None
        Hard truncation. None = no limit.
    """

    def __init__(
        self,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = 1024,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_length = max_length
        self.vocab_size = VOCAB_SIZE
        self.pad_id     = PAD_ID
        self.mask_id    = MASK_ID
        self.bos_id     = BOS_ID
        self.eos_id     = EOS_ID
        self.unk_id     = UNK_ID

    # ── Core encode / decode ─────────────────────────────────────────────────

    def encode(self, sequence: str) -> List[int]:
        """Convert an amino-acid string to a list of token ids."""
        sequence = sequence.upper().strip()
        ids: List[int] = []
        if self.add_bos:
            ids.append(BOS_ID)
        for aa in sequence:
            ids.append(_TOKEN2ID.get(aa, UNK_ID))
        if self.add_eos:
            ids.append(EOS_ID)
        if self.max_length is not None:
            ids = ids[: self.max_length]
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special: bool = True) -> str:
        """Convert token ids back to an amino-acid string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = [_ID2TOKEN.get(i, "<UNK>") for i in ids]
        if skip_special:
            special_set = set(_SPECIAL)
            tokens = [t for t in tokens if t not in special_set]
        return "".join(tokens)

    # ── Batch helpers ────────────────────────────────────────────────────────

    def batch_encode(
        self,
        sequences: List[str],
        pad: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of sequences into a padded LongTensor.

        Returns
        -------
        torch.LongTensor of shape (B, max_len)
        """
        encoded = [self.encode(s) for s in sequences]
        if pad:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [PAD_ID] * (max_len - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)

    def batch_decode(self, ids: torch.Tensor, skip_special: bool = True) -> List[str]:
        return [self.decode(row, skip_special=skip_special) for row in ids]

    # ── Convenience ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return VOCAB_SIZE

    def get_vocab(self) -> dict:
        return dict(_TOKEN2ID)
