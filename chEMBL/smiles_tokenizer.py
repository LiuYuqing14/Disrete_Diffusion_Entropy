"""
smiles_tokenizer.py
-------------------
Character-level SMILES tokenizer for Bio-SEDD (ChEMBL).

Design notes
------------
SMILES (Simplified Molecular Input Line Entry System) uses a character-level
alphabet, but it is NOT purely single-character — some atoms are two characters:
  Cl, Br, Si, Se, Te, @@, [nH], [NH], etc.

We handle this with a greedy longest-match tokeniser that recognises all
two-character atoms before falling back to single characters.

Vocabulary breakdown (size = 94):
  ID  0  : <PAD>  – padding
  ID  1  : <MASK> – absorbing token (SEDD absorb graph)
  ID  2  : <BOS>  – begin of sequence (optional)
  ID  3  : <EOS>  – end of sequence   (optional)
  ID  4  : <UNK>  – unknown character
  ID 5+  : SMILES characters (atoms, bonds, brackets, ring closures, ...)

The full character set covers all tokens seen in ChEMBL-33 / ZINC250k.
"""

from __future__ import annotations
import re
from typing import List, Union
import torch

# ── Two-character atoms (must be matched BEFORE single chars) ─────────────────
_TWO_CHAR = [
    "Cl", "Br", "Si", "Se", "Te",
    "@@", "[nH]", "[NH]", "[NH2]", "[NH3]", "[NH4]",
    "[OH]", "[OH2]", "[SH]", "[S@@]", "[S@]",
    "[C@@H]", "[C@H]", "[C@@]", "[C@]",
    "[N+]", "[N-]", "[O+]", "[O-]", "[S+]", "[S-]",
    "[n+]", "[n-]", "[o+]", "[Na+]", "[K+]", "[Ca+2]",
    "[Mg+2]", "[Zn+2]", "[Fe+2]", "[Fe+3]",
    "[2H]", "[13C]", "[15N]",
]

# ── Single-character SMILES tokens ────────────────────────────────────────────
_SINGLE = list("CNOSPFIBcnops=#@+\\/-[]()%0123456789.")

# ── Build ordered vocab (two-char first for greedy match) ─────────────────────
_SPECIALS = ["<PAD>", "<MASK>", "<BOS>", "<EOS>", "<UNK>"]
_SMILES_TOKENS = _TWO_CHAR + _SINGLE

# Deduplicate while preserving order
_seen: set = set()
_UNIQUE_SMILES: List[str] = []
for t in _SMILES_TOKENS:
    if t not in _seen:
        _UNIQUE_SMILES.append(t)
        _seen.add(t)

_VOCAB: List[str] = _SPECIALS + _UNIQUE_SMILES
_TOKEN2ID = {tok: idx for idx, tok in enumerate(_VOCAB)}
_ID2TOKEN  = {idx: tok for tok, idx in _TOKEN2ID.items()}

# Convenient constants
PAD_ID     = 0
MASK_ID    = 1
BOS_ID     = 2
EOS_ID     = 3
UNK_ID     = 4
VOCAB_SIZE = len(_VOCAB)

# Build regex for tokenisation (longest match first)
_sorted_tokens = sorted(_UNIQUE_SMILES, key=len, reverse=True)
_PATTERN = re.compile(
    "(" + "|".join(re.escape(t) for t in _sorted_tokens) + ")"
)


class SmilesTokenizer:
    """
    Greedy longest-match character-level SMILES tokenizer.

    Parameters
    ----------
    add_bos      : prepend <BOS>
    add_eos      : append  <EOS>
    max_length   : hard truncation (None = no limit)
    canonicalize : if True, use RDKit to canonicalize before encoding
                   (requires rdkit; gracefully skips if not installed)
    """

    def __init__(
        self,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = 128,
        canonicalize: bool = False,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_length = max_length
        self.canonicalize = canonicalize
        self.vocab_size = VOCAB_SIZE
        self.pad_id  = PAD_ID
        self.mask_id = MASK_ID
        self.bos_id  = BOS_ID
        self.eos_id  = EOS_ID
        self.unk_id  = UNK_ID

        self._rdkit_available = False
        if canonicalize:
            try:
                from rdkit import Chem  # noqa: F401
                self._rdkit_available = True
            except ImportError:
                print("[SmilesTokenizer] RDKit not found; canonicalization disabled.")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _canonicalize(self, smi: str) -> str:
        if not self._rdkit_available:
            return smi
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol is not None else smi

    @staticmethod
    def _tokenize(smi: str) -> List[str]:
        """Split a SMILES string into a list of tokens."""
        tokens: List[str] = []
        pos = 0
        while pos < len(smi):
            m = _PATTERN.match(smi, pos)
            if m:
                tokens.append(m.group(0))
                pos = m.end()
            else:
                tokens.append(smi[pos])  # unknown char → <UNK>
                pos += 1
        return tokens

    # ── Core encode / decode ─────────────────────────────────────────────────

    def encode(self, smiles: str) -> List[int]:
        if self.canonicalize:
            smiles = self._canonicalize(smiles)
        tokens = self._tokenize(smiles.strip())
        ids: List[int] = []
        if self.add_bos:
            ids.append(BOS_ID)
        for tok in tokens:
            ids.append(_TOKEN2ID.get(tok, UNK_ID))
        if self.add_eos:
            ids.append(EOS_ID)
        if self.max_length is not None:
            ids = ids[: self.max_length]
        return ids

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special: bool = True,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special_ids = {PAD_ID, MASK_ID, BOS_ID, EOS_ID} if skip_special else set()
        parts = []
        for i in ids:
            if i in special_ids:
                continue
            parts.append(_ID2TOKEN.get(i, "?"))
        return "".join(parts)

    # ── Batch helpers ────────────────────────────────────────────────────────

    def batch_encode(
        self,
        smiles_list: List[str],
        pad: bool = True,
    ) -> torch.Tensor:
        encoded = [self.encode(s) for s in smiles_list]
        if pad:
            max_len = max(len(e) for e in encoded)
            encoded = [
                e + [PAD_ID] * (max_len - len(e)) for e in encoded
            ]
        return torch.tensor(encoded, dtype=torch.long)

    def batch_decode(
        self,
        ids: torch.Tensor,
        skip_special: bool = True,
    ) -> List[str]:
        return [self.decode(row, skip_special=skip_special) for row in ids]

    def __len__(self) -> int:
        return VOCAB_SIZE

    def get_vocab(self) -> dict:
        return dict(_TOKEN2ID)
