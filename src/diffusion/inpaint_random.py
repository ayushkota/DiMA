"""
Infilling of randomly masked residues with the pretrained DiMA (no training, no adapter).

For every test antibody:
  1. pick 15% of its residues (scattered positions, or one contiguous stretch), replace them by ESM2's <mask> token,
     encode with ESM2 and keep the rows of the unmasked residues as clean latents c;
  2. start the masked rows from pure noise and run the usual reverse-diffusion loop;
  3. after every step overwrite the unmasked rows with mu(t) * c + std(t) * fresh_noise (same as inpaint_baseline);
  4. decode and compare the generated letters with the real ones at the masked positions.

Example of usage:
python -m src.diffusion.inpaint_random --config_path="src/configs" --n 500 --offline --mask_mode scattered
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk

from src.diffusion.inpaint_baseline import AA, aa_freq, inpaint, load_model, tv_distance
from src.utils.hydra_utils import setup_config


def make_mask(lengths, frac, mode, rng, n_tokens):
    """Boolean [B, n_tokens] mask over ESM2 token positions. Token i+1 is residue i, because <cls> sits at position 0."""
    mask = torch.zeros(len(lengths), n_tokens, dtype=torch.bool)
    for b, L in enumerate(lengths):
        k = max(1, round(frac * L))
        if mode == "scattered":
            idx = rng.choice(L, size=k, replace=False)
        else:  # one contiguous stretch
            s = int(rng.integers(0, L - k + 1))
            idx = np.arange(s, s + k)
        mask[b, torch.as_tensor(idx) + 1] = True
    return mask


@torch.no_grad()
def encode_random_masked(model, seqs, frac, mode, rng, max_sequence_len):
    """Like encode_masked in inpaint_baseline, but the masked positions are random. Returns latents c,
    the attention mask and the boolean mask of the masked token positions."""
    enc, device = model.encoder, model.device
    tok = enc.tokenizer(
        seqs, return_tensors="pt", padding=True, truncation=True,
        max_length=max_sequence_len + 2, return_attention_mask=True,
    ).to(device)
    masked = make_mask([len(s) for s in seqs], frac, mode, rng, tok["input_ids"].size(1)).to(device)
    ids = tok["input_ids"].clone()
    ids[masked] = enc.tokenizer.mask_token_id
    hidden = enc.encoder(input_ids=ids, attention_mask=tok["attention_mask"]).last_hidden_state.to(device)
    c = enc.enc_normalizer.normalize(hidden)
    return c, tok["attention_mask"].int(), masked


def summarize(df, ref_seqs):
    freq = aa_freq(ref_seqs)
    gen_masked = "".join(df["masked_gen"])
    real_masked = "".join(df["masked_true"])
    gen_freq = np.array([gen_masked.count(a) for a in AA], dtype=float)
    gen_freq /= gen_freq.sum()
    real_freq = np.array([real_masked.count(a) for a in AA], dtype=float)
    real_freq /= real_freq.sum()
    return {
        "n": len(df),
        "masked_residues_total": len(real_masked),
        "recovery_masked_positions": float((np.array(list(gen_masked)) == np.array(list(real_masked))).mean()),
        "chance_draw_from_composition": float((freq ** 2).sum()),
        "chance_always_most_common_letter": float(freq.max()),
        "composition_TV_generated_vs_real_masked": tv_distance(gen_freq, real_freq),
        "unmasked_positions_intact": float(df["unmasked_ok"].mean()),
        "length_mismatches": int(df["length_mismatch"].sum()),
    }


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

    n_steps = args.n_steps or config.generation.N_steps
    t_min = config.generation.t_min
    model = load_model(config, device)

    ds_full = load_from_disk(f"{args.data_dir}/{args.split}")
    ref_seqs = ds_full["sequence"]
    ds = ds_full.select(range(min(args.n, len(ds_full))))
    print(f"Inpainting {args.mask_frac:.0%} random residues ({args.mask_mode}) of {len(ds)} antibodies, {n_steps} steps",
          flush=True)

    rows = []
    for start in range(0, len(ds), args.batch_size):
        seqs = ds[start:start + args.batch_size]["sequence"]
        c, attention_mask, masked = encode_random_masked(model, seqs, args.mask_frac, args.mask_mode, rng,
                                                        args.max_sequence_len)
        x_final = inpaint(model, c, attention_mask, masked, n_steps, t_min, selfcond_framework=False)
        decoded = model.encoder.batch_decode(x_final, attention_mask=attention_mask)

        masked = masked.cpu().numpy()
        for b, (seq, dec) in enumerate(zip(seqs, decoded)):
            pos = np.flatnonzero(masked[b]) - 1  # residue indices
            if len(dec) != len(seq):
                rows.append({"sequence": seq, "masked_positions": " ".join(map(str, pos)), "masked_true": "",
                             "masked_gen": "", "unmasked_ok": False, "length_mismatch": True})
                continue
            keep = np.ones(len(seq), dtype=bool)
            keep[pos] = False
            rows.append({
                "sequence": seq,
                "masked_positions": " ".join(map(str, pos)),
                "masked_true": "".join(seq[p] for p in pos),
                "masked_gen": "".join(dec[p] for p in pos),
                "unmasked_ok": all(dec[i] == seq[i] for i in np.flatnonzero(keep)),
                "length_mismatch": False,
            })
        print(f"  {len(rows)}/{len(ds)}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, f"{args.tag}.csv"), index=False)

    summary = summarize(df[~df["length_mismatch"]], ref_seqs)
    summary["length_mismatches"] = int(df["length_mismatch"].sum())
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Random {args.mask_frac:.0%} masking ({args.mask_mode}) with the pretrained DiMA ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nExamples (masked real -> generated):")
    for _, r in df[~df["length_mismatch"]].head(10).iterrows():
        print(f"  {r['masked_true']} -> {r['masked_gen']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="results/inpaint_random")
    parser.add_argument("--tag", type=str, default=None, help="default: random15_<mask_mode>")
    parser.add_argument("--mask_mode", type=str, default="scattered", choices=["scattered", "span"])
    parser.add_argument("--mask_frac", type=float, default=0.15)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=None, help="default: generation.N_steps from the config (2000)")
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.tag = args.tag or f"random{round(args.mask_frac * 100)}_{args.mask_mode}"
    main(args)
