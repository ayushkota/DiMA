"""
Build a small heavy-chain OAS subset in DiMA's dataset format.

Steps: keep heavy chains, join the 7 regions into one sequence, drop sequences with
non-standard residues, remove duplicates (also across splits), sample, and save.

Example of usage:
python -m src.datasets.prepare_oas --oas_dir oasubset --out_dir data/oas
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

REGIONS = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
META = ["File_ID", "v_call", "j_call"]
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def process_chunk(df: pd.DataFrame, n_candidates: int, seed: int) -> pd.DataFrame:
    """Heavy chains only; build the full sequence and the CDR-H3 position."""
    df = df[df["locus"] == "H"].dropna(subset=REGIONS)
    # sample first (with margin for the rows dropped below) so the string work stays cheap
    df = df.sample(n=min(n_candidates * 2, len(df)), random_state=seed).copy()
    df["sequence"] = df[REGIONS[0]]
    for r in REGIONS[1:]:
        df["sequence"] = df["sequence"] + df[r]
    df = df[df["sequence"].str.fullmatch(f"[{STANDARD_AA}]+")]
    # CDR-H3 occupies [cdr3_start, cdr3_end) in the full sequence
    df["cdr3_start"] = df[REGIONS[:5]].apply(lambda col: col.str.len()).sum(axis=1)
    df["cdr3_end"] = df["cdr3_start"] + df["cdr3_aa"].str.len()
    df["File_ID"] = df["File_ID"].astype(str)
    return df[["sequence", "cdr3_start", "cdr3_end", "cdr3_aa"] + META].drop_duplicates("sequence")


def sample_split(path: str, n_target: int, row_groups, rng, oversample: float = 1.5) -> pd.DataFrame:
    """Take roughly the same number of candidates from every row group, then dedup and trim."""
    pf = pq.ParquetFile(path)
    groups = list(range(pf.metadata.num_row_groups)) if row_groups is None else row_groups
    per_group = int(np.ceil(n_target * oversample / len(groups)))
    cols = ["locus"] + REGIONS + META
    parts = []
    for i, g in enumerate(groups):
        seed = int(rng.integers(1 << 31))
        chunk = process_chunk(pf.read_row_group(g, columns=cols).to_pandas(), per_group, seed)
        parts.append(chunk.sample(n=min(per_group, len(chunk)), random_state=seed))
        print(f"  {os.path.basename(path)} row group {g} ({i + 1}/{len(groups)}): kept {len(parts[-1])}", flush=True)
    df = pd.concat(parts).drop_duplicates("sequence")
    return df.sample(n=min(n_target, len(df)), random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


def main(args):
    rng = np.random.default_rng(args.seed)
    test = sample_split(os.path.join(args.oas_dir, "testing.parquet"), args.n_test, None, rng)
    val = sample_split(os.path.join(args.oas_dir, "validation.parquet"), args.n_val, None, rng)
    train = sample_split(os.path.join(args.oas_dir, "training.parquet"), args.n_train, None, rng)

    # no identical sequence may appear in more than one split
    val = val[~val["sequence"].isin(test["sequence"])]
    train = train[~train["sequence"].isin(set(test["sequence"]) | set(val["sequence"]))]

    dt = DatasetDict({
        "train": Dataset.from_pandas(train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val.reset_index(drop=True)),
        "test": Dataset.from_pandas(test.reset_index(drop=True)),
    })
    dt.save_to_disk(args.out_dir)

    for name, df in [("train", train), ("validation", val), ("test", test)]:
        print(f"{name}: {len(df)} sequences | length min/median/max "
              f"{df['sequence'].str.len().min()}/{int(df['sequence'].str.len().median())}/{df['sequence'].str.len().max()} "
              f"| CDR-H3 median length {int(df['cdr3_aa'].str.len().median())} "
              f"| distinct File_IDs {df['File_ID'].nunique()}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oas_dir", type=str, default="oasubset")
    parser.add_argument("--out_dir", type=str, default="data/oas")
    parser.add_argument("--n_train", type=int, default=200_000)
    parser.add_argument("--n_val", type=int, default=10_000)
    parser.add_argument("--n_test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
