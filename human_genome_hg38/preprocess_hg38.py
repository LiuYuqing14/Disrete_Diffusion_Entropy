"""
preprocess_hg38.py  —  3-mer tokenization version
====================================================
Changes vs original (single-base vocab=5):
  - vocab size: 5  →  65  (64 valid 3-mers + 1 "contains-N" token)
  - chunk size:  1024 bases  →  3072 bases  (= 1024 non-overlapping 3-mers)
  - each saved token represents one 3-mer, NOT one base
  - N-filtering still done at the base level before tokenization

Output:  data/hg38_3mer_tokens.npy   shape (N, 1024)  dtype=uint16
         data/kmer_vocab.json         index → 3-mer string  (for decoding)

Usage:
    pip install biopython
    python preprocess_hg38.py --fasta data/hg38.fa.gz --out data/
"""

import argparse
import itertools
import json
import os
import numpy as np
from Bio import SeqIO
import gzip

# ── Build 3-mer vocabulary ────────────────────────────────────────────────────
BASES = ['A', 'T', 'G', 'C']
ALL_3MERS = [''.join(t) for t in itertools.product(BASES, repeat=3)]  # 64 entries

KMER2IDX = {kmer: i for i, kmer in enumerate(ALL_3MERS)}   # e.g. "ATG" → 3
N_TOKEN  = len(ALL_3MERS)                                   # index 64 = "contains N"
VOCAB_SIZE = N_TOKEN + 1                                     # 65 total

IDX2KMER = {v: k for k, v in KMER2IDX.items()}
IDX2KMER[N_TOKEN] = '[N]'

# ── Hyperparameters ───────────────────────────────────────────────────────────
K           = 3           # k-mer length
SEQ_TOKENS  = 1024        # tokens per training sample
SEQ_BASES   = SEQ_TOKENS * K   # = 3072 bases per chunk
N_THRESHOLD = 0.1         # skip chunk if > 10% bases are N


def encode_chunk(bases: str) -> np.ndarray:
    """
    Convert a string of 3*SEQ_TOKENS bases into SEQ_TOKENS integer tokens.
    Any k-mer that contains an N is mapped to N_TOKEN (64).
    """
    tokens = np.empty(SEQ_TOKENS, dtype=np.uint16)
    for i in range(SEQ_TOKENS):
        kmer = bases[i*K : i*K + K]
        if 'N' in kmer:
            tokens[i] = N_TOKEN
        else:
            tokens[i] = KMER2IDX[kmer]
    return tokens


def process_fasta(fasta_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    chunks = []
    skipped_n = 0
    skipped_short = 0

    opener = gzip.open if fasta_path.endswith('.gz') else open
    mode   = 'rt' if fasta_path.endswith('.gz') else 'r'

    print(f"Reading {fasta_path} ...")
    with opener(fasta_path, mode) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq).upper()
            chrom_len = len(seq)
            print(f"  {record.id}  ({chrom_len:,} bp)", end='', flush=True)

            n_added = 0
            # Non-overlapping windows of SEQ_BASES bases
            for start in range(0, chrom_len - SEQ_BASES + 1, SEQ_BASES):
                window = seq[start : start + SEQ_BASES]

                # Base-level N filter
                n_frac = window.count('N') / SEQ_BASES
                if n_frac > N_THRESHOLD:
                    skipped_n += 1
                    continue

                chunks.append(encode_chunk(window))
                n_added += 1

            print(f"  →  {n_added:,} chunks")

    print(f"\nTotal chunks : {len(chunks):,}")
    print(f"Skipped (>N) : {skipped_n:,}")

    arr = np.stack(chunks)    # shape: (N, 1024)
    npy_path = os.path.join(out_dir, 'hg38_3mer_tokens.npy')
    np.save(npy_path, arr)
    print(f"Saved tokens : {npy_path}  {arr.shape}  dtype={arr.dtype}")

    # Save vocab for decoding
    vocab_path = os.path.join(out_dir, 'kmer_vocab.json')
    with open(vocab_path, 'w') as vf:
        json.dump(IDX2KMER, vf, indent=2)
    print(f"Saved vocab  : {vocab_path}  ({VOCAB_SIZE} entries)")


def decode_tokens(tokens: np.ndarray, vocab_path: str = None) -> str:
    """
    Convenience: convert a (SEQ_TOKENS,) array back to a base string.
    If a token is N_TOKEN (64), outputs 'NNN'.
    """
    if vocab_path:
        with open(vocab_path) as f:
            idx2kmer = {int(k): v for k, v in json.load(f).items()}
    else:
        idx2kmer = IDX2KMER
        idx2kmer[N_TOKEN] = 'NNN'

    return ''.join(idx2kmer.get(int(t), 'NNN') for t in tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default='data/hg38.fa.gz',
                        help='Path to hg38 FASTA (plain or gzipped)')
    parser.add_argument('--out',   default='data/',
                        help='Output directory')
    args = parser.parse_args()

    print(f"Vocab size : {VOCAB_SIZE}  ({len(ALL_3MERS)} 3-mers + 1 N-token)")
    print(f"Chunk size : {SEQ_BASES} bases  =  {SEQ_TOKENS} tokens\n")
    process_fasta(args.fasta, args.out)