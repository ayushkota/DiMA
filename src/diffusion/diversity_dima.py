"""
How diverse is DiMA's infilling for one and the same masked antibody?

Takes the first batch of antibodies with exactly the masks of inpaint_random, then runs the infilling several times from
different noise. Reports the recovery per sample and the fraction of masked positions where two samples differ.
Compare with the sampled diversity of ESM2 in esm2_masked_prediction.

Example of usage:
python -m src.diffusion.diversity_dima --config_path="src/configs" --offline --mask_mode scattered
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_from_disk

from src.diffusion.inpaint_baseline import inpaint, load_model
from src.diffusion.inpaint_random import encode_random_masked
from src.utils.hydra_utils import setup_config


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
    n_steps, t_min = config.generation.N_steps, config.generation.t_min
    model = load_model(config, device)

    ds = load_from_disk(f"{args.data_dir}/{args.split}").select(range(args.n))
    seqs = ds["sequence"]
    c, attention_mask, masked = encode_random_masked(model, seqs, args.mask_frac, args.mask_mode, rng,
                                                    args.max_sequence_len)
    masked_np = masked.cpu().numpy()

    samples = []  # per run: list over antibodies of the letters at the masked positions
    for k in range(args.samples):
        x_final = inpaint(model, c, attention_mask, masked, n_steps, t_min, selfcond_framework=False)
        decoded = model.encoder.batch_decode(x_final, attention_mask=attention_mask)
        samples.append(decoded)
        print(f"  sample {k + 1}/{args.samples}", flush=True)

    true, gen = [], [[] for _ in range(args.samples)]
    for b, seq in enumerate(seqs):
        if any(len(samples[k][b]) != len(seq) for k in range(args.samples)):
            continue
        pos = np.flatnonzero(masked_np[b]) - 1
        true += [seq[p] for p in pos]
        for k in range(args.samples):
            gen[k] += [samples[k][b][p] for p in pos]
    true, gen = np.array(true), np.array(gen)  # gen: [samples, positions]
    pairs = [(i, j) for i in range(args.samples) for j in range(i + 1, args.samples)]
    summary = {
        "mask_mode": args.mask_mode,
        "antibodies": args.n,
        "samples_per_antibody": args.samples,
        "masked_residues_total": int(len(true)),
        "recovery_per_sample": [float((gen[k] == true).mean()) for k in range(args.samples)],
        "diversity (fraction of positions where two independent samples differ)":
            float(np.mean([(gen[i] != gen[j]).mean() for i, j in pairs])),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"diversity_dima_{args.mask_mode}.json"), "w") as f:
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
    parser.add_argument("--n", type=int, default=50, help="one batch, so the masks equal the first batch of inpaint_random")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
