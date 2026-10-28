"""
Duplicate check for the OAS heavy-chain parquet files (training / validation / testing).

For each file (heavy chains only) it reports:
  - exact duplicates: rows whose full sequence (all 7 regions joined) appears more than once in the same file
  - exact overlap between files: rows of validation/testing whose full sequence also appears in training
  - near duplicates: rows that share a "clonotype key" (same V gene, same J gene, identical CDR3) with training
    but are not identical in full. For a sample of them, the number of differing residues to the closest
    training sequence of the same length is counted.

Sequences are compared through 64-bit hashes (collision chance is negligible at this size).

Example of usage:
python -m src.datasets.check_duplicates --oas_dir oasubset --out results/duplicate_check.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REGIONS = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
COLUMNS = ["locus"] + REGIONS + ["v_call", "j_call"]


def hash_strings(s: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(s, index=False).to_numpy()


def prepare_chunk(df: pd.DataFrame):
    """Heavy chains of one chunk -> (full sequences, hash of full sequence, hash of CDR3, hash of clonotype key)."""
    n_rows = len(df)
    df = df[df["locus"] == "H"].dropna(subset=REGIONS)
    seq = df[REGIONS[0]].astype(object)
    for r in REGIONS[1:]:
        seq = seq + df[r].astype(object)
    v = df["v_call"].astype(object).fillna("").str.split("*").str[0]
    j = df["j_call"].astype(object).fillna("").str.split("*").str[0]
    clono = v + "|" + j + "|" + df["cdr3_aa"].astype(object)
    return n_rows, seq, hash_strings(seq), hash_strings(df["cdr3_aa"].astype(object)), hash_strings(clono), hash_strings(clono)


def read_file(path, name, sample_size=0, rng=None):
    """Hash every heavy chain of a parquet file. Optionally pick a random sample of sequences to look at closely."""
    pf = pq.ParquetFile(path)
    n_rows, full, cdr3, clono = 0, [], [], []
    sample = {}  # clonotype hash -> list of sequences of the sampled rows
    for g in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(g, columns=COLUMNS).to_pandas()
        rows, seq, h_full, h_cdr3, h_clono, _ = prepare_chunk(df)
        n_rows += rows
        full.append(h_full); cdr3.append(h_cdr3); clono.append(h_clono)
        if sample_size:
            k = min(sample_size // pf.metadata.num_row_groups + 1, len(seq))
            for i in rng.choice(len(seq), size=k, replace=False):
                sample.setdefault(int(h_clono[i]), []).append(seq.iloc[i])
        print(f"  {name} chunk {g + 1}/{pf.metadata.num_row_groups}", flush=True)
    return n_rows, np.concatenate(full), np.concatenate(cdr3), np.concatenate(clono), sample


def within_file_stats(n_rows, full, cdr3, clono):
    u_full = np.unique(full)
    return {
        "rows": int(n_rows),
        "heavy_rows": int(len(full)),
        "unique_full_sequences": int(len(u_full)),
        "exact_duplicate_rows": int(len(full) - len(u_full)),
        "exact_duplicate_fraction": float(1 - len(u_full) / len(full)),
        "unique_cdr3": int(len(np.unique(cdr3))),
        "unique_clonotype_keys": int(len(np.unique(clono))),
    }, u_full


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def main(args):
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    d = args.oas_dir
    data, report = {}, {}
    for name, fname, sample in [("validation", "validation.parquet", args.sample), ("testing", "testing.parquet", args.sample),
                                ("training", "training.parquet", 0)]:
        print(f"Reading {fname}", flush=True)
        data[name] = read_file(os.path.join(d, fname), name, sample, rng)
        stats, u_full = within_file_stats(*data[name][:4])
        report[name] = stats
        data[name] = data[name] + (u_full,)
        print(json.dumps(stats), flush=True)

    train_full, train_clono = data["training"][1], data["training"][3]
    u_train_full, u_train_clono = data["training"][5], np.unique(train_clono)

    # overlap with training and between validation and testing, counted over rows
    for name in ("validation", "testing"):
        _, full, _, clono, _, _ = data[name]
        exact = np.isin(full, u_train_full)
        same_clono = np.isin(clono, u_train_clono)
        report[name]["rows_identical_to_a_training_sequence"] = int(exact.sum())
        report[name]["fraction_identical_to_a_training_sequence"] = float(exact.mean())
        report[name]["rows_near_duplicate_of_training (same clonotype key, not identical)"] = int((same_clono & ~exact).sum())
        report[name]["fraction_near_duplicate_of_training"] = float((same_clono & ~exact).mean())
    report["validation_vs_testing_identical_rows"] = int(np.isin(data["testing"][1], data["validation"][5]).sum())

    # how different are the near duplicates? look at a sample, scanning training once more
    wanted = {h for name in ("validation", "testing") for h in data[name][4]}
    print(f"Second pass over training for {len(wanted)} sampled clonotype keys", flush=True)
    pf = pq.ParquetFile(os.path.join(d, "training.parquet"))
    matches = {}
    wanted_arr = np.array(sorted(wanted), dtype=np.uint64)
    for g in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(g, columns=COLUMNS).to_pandas()
        _, seq, _, _, h_clono, _ = prepare_chunk(df)
        hit = np.isin(h_clono, wanted_arr)
        for h, s in zip(h_clono[hit], seq[hit]):
            lst = matches.setdefault(int(h), [])
            if len(lst) < 50:
                lst.append(s)
        print(f"  training chunk {g + 1}/{pf.metadata.num_row_groups}", flush=True)

    bins = [("identical", 0, 0), ("1-2 residues differ", 1, 2), ("3-5 residues differ", 3, 5),
            ("6-10 residues differ", 6, 10), ("more than 10 differ", 11, 10 ** 6)]
    for name in ("validation", "testing"):
        counts = {b[0]: 0 for b in bins}
        n_sampled = n_with_relative = 0
        for h, seqs in data[name][4].items():
            for s in seqs:
                n_sampled += 1
                cands = [c for c in matches.get(h, []) if len(c) == len(s)]
                if not cands:
                    continue
                n_with_relative += 1
                dist = min(hamming(s, c) for c in cands)
                for label, lo, hi in bins:
                    if lo <= dist <= hi:
                        counts[label] += 1
        report[name]["closeness_sample"] = {
            "sampled_rows": n_sampled,
            "with_a_same-length_training_relative": n_with_relative,
            "closest_training_relative (residues differing)": counts,
        }

    report["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oas_dir", type=str, default="oasubset")
    parser.add_argument("--out", type=str, default="results/duplicate_check.json")
    parser.add_argument("--sample", type=int, default=3000, help="rows sampled from validation and testing for the closeness check")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
