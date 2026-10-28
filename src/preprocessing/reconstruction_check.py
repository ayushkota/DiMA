"""
Reconstruction check: encode OAS heavy-chain sequences with the frozen ESM2 encoder, decode them back
with the frozen decoder, and measure how many residues (overall and inside CDR-H3) are recovered.

Example of usage:
python -m src.preprocessing.reconstruction_check --config_path="src/configs" --n 5000
"""
import argparse

import numpy as np
import torch
from datasets import load_from_disk
from hydra.utils import instantiate

from src.utils.hydra_utils import setup_config


@torch.no_grad()
def main(args):
    config = setup_config(config_path=args.config_path)
    if args.offline:
        # compute nodes have no internet: load ESM2 from the local Hugging Face cache folder instead
        from huggingface_hub import snapshot_download
        config.encoder.config.encoder_model_name = snapshot_download(
            config.encoder.config.encoder_model_name, local_files_only=True
        )
    device = torch.device("cuda:0")

    encoder = instantiate(config.encoder)(device=device, main_config=config, add_enc_normalizer=True)
    encoder.eval()

    ds = load_from_disk(f"{args.data_dir}/{args.split}")
    ds = ds.select(range(min(args.n, len(ds))))
    print(f"Checking {len(ds)} sequences from {args.data_dir}/{args.split}", flush=True)

    n_res = n_ok = n_cdr = n_cdr_ok = 0
    n_seq_exact = n_cdr_exact = n_len_mismatch = 0
    examples = []

    for start in range(0, len(ds), args.batch_size):
        batch = ds[start:start + args.batch_size]
        encodings, attention_mask, _ = encoder.batch_encode(batch, max_sequence_len=args.max_sequence_len)
        decoded = encoder.batch_decode(encodings, attention_mask=attention_mask)

        for seq, pred, s, e in zip(batch["sequence"], decoded, batch["cdr3_start"], batch["cdr3_end"]):
            if len(pred) != len(seq):
                n_len_mismatch += 1
                n_res += len(seq)
                n_cdr += e - s
                continue
            match = np.array([a == b for a, b in zip(seq, pred)])
            n_res += len(seq)
            n_ok += match.sum()
            n_cdr += e - s
            n_cdr_ok += match[s:e].sum()
            n_seq_exact += match.all()
            n_cdr_exact += match[s:e].all()
            if not match.all() and len(examples) < 5:
                examples.append((seq[s:e], pred[s:e]))

        if (start // args.batch_size) % 10 == 0:
            print(f"  {start + len(batch)}/{len(ds)}", flush=True)

    n = len(ds)
    print("\n=== Reconstruction results ===")
    print(f"sequences: {n}")
    print(f"residue accuracy (whole sequence): {n_ok / n_res:.4%}")
    print(f"residue accuracy (CDR-H3 only):    {n_cdr_ok / n_cdr:.4%}")
    print(f"exact sequence match:              {n_seq_exact / n:.4%}")
    print(f"exact CDR-H3 match:                {n_cdr_exact / n:.4%}")
    print(f"length mismatches:                 {n_len_mismatch}")
    if examples:
        print("\nExamples of CDR-H3 errors (original -> decoded):")
        for orig, pred in examples:
            print(f"  {orig} -> {pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--offline", action="store_true")
    main(parser.parse_args())
