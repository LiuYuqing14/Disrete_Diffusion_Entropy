"""
prepare_chembl.py
-----------------
Download and preprocess ChEMBL data for Bio-SEDD fine-tuning.

This script handles three data sources:

  Source A — ChEMBL web resource client (official, recommended)
    Downloads small-molecule compounds with SMILES + properties.
    Requires: pip install chembl_webresource_client

  Source B — ChEMBL SQLite dump (bulk download, fastest for large runs)
    Download from: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
    File: chembl_33_sqlite.tar.gz (~4 GB)

  Source C — ZINC250k / GuacaMol benchmark (smallest, good for quick tests)
    Public benchmark datasets, no authentication needed.

Output: a filtered CSV with columns:
    canonical_smiles, heavy_atom_count, mol_weight, logP, QED

Usage
-----
  # Quick start with ZINC250k (no account needed, ~250k molecules):
  python prepare_chembl.py --source zinc250k --output /data/zinc250k.csv

  # ChEMBL via API (requires internet, downloads up to --max_mols molecules):
  python prepare_chembl.py --source api --max_mols 500000 --output /data/chembl.csv

  # ChEMBL from local SQLite:
  python prepare_chembl.py --source sqlite \
      --sqlite_path /data/chembl_33.db \
      --output /data/chembl_filtered.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Source A: ChEMBL web API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_api(max_mols: int, output_path: str) -> None:
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("Install with: pip install chembl_webresource_client")
        sys.exit(1)

    print(f"Fetching up to {max_mols:,} molecules from ChEMBL API ...")
    molecule = new_client.molecule

    # Filter: small organic molecules with SMILES, MW 100-600
    mols = molecule.filter(
        molecule_type="Small molecule",
        structure_type="MOL",
    ).only([
        "molecule_chembl_id",
        "molecule_structures",
        "molecule_properties",
    ])

    records = []
    for i, mol in enumerate(mols):
        if i >= max_mols:
            break
        if i % 10000 == 0:
            print(f"  {i:,} fetched ...", end="\r")

        try:
            smi = mol["molecule_structures"]["canonical_smiles"]
            props = mol.get("molecule_properties") or {}
            records.append({
                "canonical_smiles": smi,
                "chembl_id": mol["molecule_chembl_id"],
                "mol_weight": props.get("full_mwt"),
                "logP": props.get("alogp"),
                "hbd": props.get("hbd"),
                "hba": props.get("hba"),
                "psa": props.get("psa"),
                "ro5_violations": props.get("num_ro5_violations"),
            })
        except (KeyError, TypeError):
            continue

    print(f"\nFetched {len(records):,} molecules")
    _save_filtered(records, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Source B: local SQLite
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_sqlite(sqlite_path: str, output_path: str, max_mols: int) -> None:
    import sqlite3

    print(f"Reading from SQLite: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)

    query = f"""
        SELECT
            cs.canonical_smiles,
            md.chembl_id,
            cp.full_mwt      AS mol_weight,
            cp.alogp         AS logP,
            cp.hbd,
            cp.hba,
            cp.psa,
            cp.num_ro5_violations AS ro5_violations
        FROM compound_structures cs
        JOIN molecule_dictionary md ON cs.molregno = md.molregno
        LEFT JOIN compound_properties cp ON cs.molregno = cp.molregno
        WHERE cs.canonical_smiles IS NOT NULL
          AND md.molecule_type = 'Small molecule'
        LIMIT {max_mols}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Read {len(df):,} rows from SQLite")
    _save_filtered(df.to_dict("records"), output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Source C: ZINC250k / GuacaMol
# ─────────────────────────────────────────────────────────────────────────────

ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

GUACAMOL_TRAIN_URL = (
    "https://figshare.com/ndownloader/files/13612760"  # guacamol_v1_train.smiles
)

def fetch_zinc250k(output_path: str) -> None:
    print("Downloading ZINC250k ...")
    try:
        df = pd.read_csv(ZINC250K_URL)
    except Exception:
        print(
            "Direct download failed. Download manually from:\n"
            f"  {ZINC250K_URL}\n"
            "and pass as --source csv --input_csv zinc250k.csv"
        )
        sys.exit(1)

    # Rename column to match our convention
    if "smiles" in df.columns:
        df = df.rename(columns={"smiles": "canonical_smiles"})

    print(f"Downloaded {len(df):,} molecules")
    _save_filtered(df.to_dict("records"), output_path)


def fetch_from_csv(input_csv: str, output_path: str) -> None:
    print(f"Loading from CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    # Normalise SMILES column name
    for col in ["smiles", "SMILES", "Smiles", "structure"]:
        if col in df.columns:
            df = df.rename(columns={col: "canonical_smiles"})
            break
    print(f"Loaded {len(df):,} rows")
    _save_filtered(df.to_dict("records"), output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Shared filter + save
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rdkit_props(smi: str) -> Optional[dict]:
    """Compute QED, logP, heavy atom count via RDKit. Returns None if invalid."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return {
            "canonical_smiles": Chem.MolToSmiles(mol),  # canonicalize
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "QED": round(QED.qed(mol), 4),
            "rdkit_logP": round(Descriptors.MolLogP(mol), 4),
            "mol_weight": round(Descriptors.MolWt(mol), 2),
        }
    except ImportError:
        return {"canonical_smiles": smi}
    except Exception:
        return None


def _save_filtered(records: List[dict], output_path: str) -> None:
    """Apply standard filters and save to CSV."""
    print("Filtering and computing RDKit properties ...")
    rows = []
    for i, rec in enumerate(records):
        if i % 50000 == 0:
            print(f"  {i:,} / {len(records):,} processed ...", end="\r")
        smi = str(rec.get("canonical_smiles", "")).strip()
        if not smi:
            continue
        props = _compute_rdkit_props(smi)
        if props is None:
            continue
        # Merge original record fields
        merged = {**rec, **props}
        rows.append(merged)

    df = pd.DataFrame(rows)

    # Standard filters (matching ChEMBLDataset defaults)
    if "heavy_atom_count" in df.columns:
        df = df[df["heavy_atom_count"].between(5, 60)]
    if "mol_weight" in df.columns:
        df = df[df["mol_weight"].between(100, 700)]

    df = df.drop_duplicates(subset=["canonical_smiles"])
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} molecules to: {output_path}")
    print(f"Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Prepare ChEMBL data for Bio-SEDD")
    p.add_argument("--source", required=True,
                   choices=["api", "sqlite", "zinc250k", "csv"],
                   help="Data source to use.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--max_mols", type=int, default=500_000,
                   help="Max molecules to download (api / sqlite sources).")
    p.add_argument("--sqlite_path", default=None,
                   help="Path to local ChEMBL SQLite file (source=sqlite).")
    p.add_argument("--input_csv", default=None,
                   help="Path to existing CSV (source=csv).")
    args = p.parse_args()

    if args.source == "api":
        fetch_from_api(args.max_mols, args.output)
    elif args.source == "sqlite":
        if not args.sqlite_path:
            print("--sqlite_path required for source=sqlite")
            sys.exit(1)
        fetch_from_sqlite(args.sqlite_path, args.output, args.max_mols)
    elif args.source == "zinc250k":
        fetch_zinc250k(args.output)
    elif args.source == "csv":
        if not args.input_csv:
            print("--input_csv required for source=csv")
            sys.exit(1)
        fetch_from_csv(args.input_csv, args.output)


if __name__ == "__main__":
    main()
