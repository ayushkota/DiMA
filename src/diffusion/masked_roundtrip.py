"""
Encoder-decoder check for the random-masking test, without any diffusion.

The unmasked rows of the final tensor in inpaint_random are the ESM2 rows of the masked sequence, passed straight to the
decoder. So decoding them here gives exactly the unmasked-residue recovery of that test, and shows whether the errors
come from ESM2 + decoder (masking shifts the neighbouring rows) or from DiMA.

Uses the same seed, batch size and masks as inpaint_random. Reports per-letter recovery of the unmasked residues
(masked encoding vs the clean encoding as a control), also by distance to the nearest masked residue.

Example of usage:
python -m src.diffusion.masked_roundtrip --config_path="src/configs" --n 500 --offline --mask_mode scattered
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_from_disk

from src.diffusion.inpaint_baseline import encode_masked, load_model
from src.diffusion.inpaint_random import encode_random_masked
from src.utils.hydra_utils import setup_config

DIST_BINS = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6+", 6, 10 ** 6)]


@torch.no_grad()
def main(args):
    config = setup_config(config_path=args.config_path)
    if args.offline:
        from huggingface_hub import snapshot_download
        config.encoder.config.encoder_model_name = snapshot_download(
            config.encoder.config.encoder_model_name, local_files_only=True
        )
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_model(config, device)

    ds = load_from_disk(f"{args.data_dir}/{args.split}")
    ds = ds.select(range(min(args.n, len(ds))))

    ok_masked = tot = ok_clean = 0
    by_dist = {b[0]: [0, 0] for b in DIST_BINS}
    seq_all_ok = n_seq = 0
    for start in range(0, len(ds), args.batch_size):
        seqs = ds[start:start + args.batch_size]["sequence"]
        c, attention_mask, masked = encode_random_masked(model, seqs, args.mask_frac, args.mask_mode, rng,
                                                        args.max_sequence_len)
        dec = model.encoder.batch_decode(c, attention_mask=attention_mask)

        zeros = torch.zeros(len(seqs), dtype=torch.long, device=device)
        c0, am0, _ = encode_masked(model, seqs, zeros, zeros, args.max_sequence_len)  # nothing masked
        dec0 = model.encoder.batch_decode(c0, attention_mask=am0)

        masked = masked.cpu().numpy()
        for b, seq in enumerate(seqs):
            if len(dec[b]) != len(seq) or len(dec0[b]) != len(seq):
                continue
            pos = np.flatnonzero(masked[b]) - 1
            keep = np.ones(len(seq), dtype=bool)
            keep[pos] = False
            right = np.array([dec[b][i] == seq[i] for i in range(len(seq))])
            right0 = np.array([dec0[b][i] == seq[i] for i in range(len(seq))])
            ok_masked += int(right[keep].sum()); ok_clean += int(right0[keep].sum()); tot += int(keep.sum())
            seq_all_ok += int(right[keep].all()); n_seq += 1
            dist = np.abs(np.arange(len(seq))[:, None] - pos[None, :]).min(1)  # distance to nearest masked residue
            for label, lo, hi in DIST_BINS:
                sel = keep & (dist >= lo) & (dist <= hi)
                by_dist[label][0] += int(right[sel].sum()); by_dist[label][1] += int(sel.sum())
        print(f"  {min(start + args.batch_size, len(ds))}/{len(ds)}", flush=True)

    summary = {
        "mask_mode": args.mask_mode,
        "n_sequences": n_seq,
        "unmasked_residues": tot,
        "per_letter_recovery_unmasked_from_masked_encoding": ok_masked / tot,
        "per_letter_recovery_unmasked_from_clean_encoding (control)": ok_clean / tot,
        "sequences_with_all_unmasked_letters_right": seq_all_ok / n_seq,
        "per_letter_recovery_by_distance_to_nearest_masked_residue": {
            k: (v[0] / v[1] if v[1] else None) for k, v in by_dist.items()},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"masked_roundtrip_{args.mask_mode}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="results/inpaint_random")
    parser.add_argument("--mask_mode", type=str, default="scattered", choices=["scattered", "span"])
    parser.add_argument("--mask_frac", type=float, default=0.15)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
