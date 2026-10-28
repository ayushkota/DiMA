"""
CDR-H3 infilling with the pretrained DiMA. Without --adapter this is the no-training baseline (approach A);
with --adapter the trained conditional adapter is used as well.

For every test antibody:
  1. mask the CDR-H3 letters, encode with ESM2 and keep the framework rows c (clean latents);
  2. start the CDR-H3 rows from pure noise and run the usual reverse-diffusion loop;
  3. after every step overwrite the framework rows with mu(t) * c + std(t) * fresh_noise
     (the known framework noised to the current noise level);
  4. decode and compare the generated CDR-H3 with the real one.

Example of usage:
python -m src.diffusion.inpaint_baseline --config_path="src/configs" --n 500 --offline --tag baseline
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk

from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.models.adapter import ConditionAdapter, ConditionedScoreEstimator
from src.utils.hydra_utils import setup_config
from src.utils.pretrained_utils import PRETRAINED_MODELS_PATHS

AA = "ACDEFGHIKLMNPQRSTVWY"
H3_END_MOTIFS = ("DY", "DI", "DV", "DP", "DL")  # typical ends from the IGHJ genes (FDY, AFDI, MDV, FDP, FDL)


def load_model(config, device):
    model = BaseDiffusionTrainer(config, device)
    paths = PRETRAINED_MODELS_PATHS[config.encoder.config.encoder_type]
    root = config.project.path
    model.restore_checkpoint(os.path.join(root, paths["diffusion"]))
    model.encoder.enc_normalizer._load_state_dict(os.path.join(root, paths["stats"]))
    model.encoder.restore_decoder(os.path.join(root, paths["decoder"]))
    model.score_estimator.eval()  # no dropout during generation
    model.encoder.eval()
    return model


@torch.no_grad()
def encode_masked(model, seqs, starts, ends, max_sequence_len):
    """Encode sequences with the CDR-H3 letters replaced by ESM2's <mask> token.
    Returns normalized latents c, the attention mask and a boolean mask of the CDR-H3 token positions."""
    enc, device = model.encoder, model.device
    tok = enc.tokenizer(
        seqs, return_tensors="pt", padding=True, truncation=True,
        max_length=max_sequence_len + 2, return_attention_mask=True,
    ).to(device)
    ids = tok["input_ids"].clone()
    pos = torch.arange(ids.size(1), device=device)[None]
    # token i+1 is residue i, because <cls> sits at position 0
    h3 = (pos >= (starts + 1)[:, None]) & (pos < (ends + 1)[:, None])
    ids[h3] = enc.tokenizer.mask_token_id
    hidden = enc.encoder(input_ids=ids, attention_mask=tok["attention_mask"]).last_hidden_state.to(device)
    c = enc.enc_normalizer.normalize(hidden)
    return c, tok["attention_mask"].int(), h3


@torch.no_grad()
def inpaint(model, c, attention_mask, h3, n_steps, t_min, selfcond_framework):
    """Reverse diffusion where only the CDR-H3 rows are generated; framework rows are re-set at every step."""
    dyn = model.dynamic
    B = c.size(0)
    framework = (attention_mask.bool() & ~h3)[..., None]

    x = torch.randn_like(c)
    x_self = torch.zeros_like(c)
    ts = torch.linspace(dyn.T, t_min, n_steps + 1, device=c.device)
    for i in range(n_steps):
        t = ts[i] * torch.ones(B, device=c.device)
        next_t = ts[i + 1] * torch.ones(B, device=c.device)
        out = model.solver.step(x_t=x, t=t, next_t=next_t, mask=attention_mask, x_0_self_cond=x_self)
        x, x_mean, x_self = out["x"], out["x_mean"], out["x_0"]

        # overwrite the framework rows with the known framework at the next noise level
        p = dyn.marginal_params(next_t)
        x = torch.where(framework, p["mu"] * c + p["std"] * torch.randn_like(c), x)
        if selfcond_framework:
            x_self = torch.where(framework, c, x_self)
    return torch.where(framework, c, x_mean)


def tv_distance(a, b):
    """Total variation distance between two amino-acid frequency vectors (0 = identical, 1 = disjoint)."""
    return 0.5 * float(np.abs(a - b).sum())


def aa_freq(seqs):
    counts = np.array([sum(s.count(a) for s in seqs) for a in AA], dtype=float)
    return counts / counts.sum()


def identity(a, b):
    return float(np.mean([x == y for x, y in zip(a, b)]))


def summarize(df, ref_h3, rng):
    gen, real = df["cdr3_gen"].tolist(), df["cdr3_true"].tolist()
    ref_freq = aa_freq(ref_h3)
    control = list(rng.choice(ref_h3, size=len(gen), replace=False))

    # recovery of the native CDR-H3, against a control that pairs each antibody with another real H3 of equal length
    by_len = {}
    for s in ref_h3:
        by_len.setdefault(len(s), []).append(s)
    ctrl_ident = []
    for r in real:
        cands = [s for s in by_len.get(len(r), []) if s != r]
        if cands:
            ctrl_ident.append(identity(r, cands[rng.integers(len(cands))]))

    summary = {
        "n": len(gen),
        "framework_roundtrip_ok": float(df["framework_ok"].mean()),
        "composition_TV_generated_vs_real": tv_distance(aa_freq(gen), ref_freq),
        "composition_TV_realsubset_vs_real (noise floor)": tv_distance(aa_freq(control), ref_freq),
        "unique_fraction_generated": len(set(gen)) / len(gen),
        "unique_fraction_real": len(set(real)) / len(real),
        "native_recovery_generated": float(np.mean([identity(r, g) for r, g in zip(real, gen)])),
        "native_recovery_control (other real H3, same length)": float(np.mean(ctrl_ident)),
        "starts_with_A_generated": float(np.mean([g.startswith("A") for g in gen])),
        "starts_with_A_real": float(np.mean([r.startswith("A") for r in real])),
        "ends_with_J_motif_generated": float(np.mean([g.endswith(H3_END_MOTIFS) for g in gen])),
        "ends_with_J_motif_real": float(np.mean([r.endswith(H3_END_MOTIFS) for r in real])),
    }
    return summary


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

    wrapper = None
    if args.adapter:
        # trained adapter: the (frozen) score estimator additionally receives the framework context
        adapter = ConditionAdapter(config.model.config).to(device)
        adapter.load_state_dict(torch.load(args.adapter, map_location="cpu")["adapter"])
        wrapper = ConditionedScoreEstimator(model.score_estimator, adapter).to(device).eval()
        model.score_estimator = wrapper  # the solver calls model.score_estimator through calc_score
        print(f"Using adapter from {args.adapter}", flush=True)

    ds_full = load_from_disk(f"{args.data_dir}/{args.split}")
    ref_h3 = ds_full["cdr3_aa"]
    ds = ds_full.select(range(min(args.n, len(ds_full))))
    print(f"Inpainting CDR-H3 of {len(ds)} antibodies, {n_steps} steps, selfcond_framework={args.selfcond_framework}", flush=True)

    rows = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start:start + args.batch_size]
        starts = torch.tensor(batch["cdr3_start"], device=device)
        ends = torch.tensor(batch["cdr3_end"], device=device)
        c, attention_mask, h3 = encode_masked(model, batch["sequence"], starts, ends, args.max_sequence_len)

        if wrapper is not None:
            wrapper.set_context(c, attention_mask, h3)
        x_final = inpaint(model, c, attention_mask, h3, n_steps, t_min, args.selfcond_framework)
        decoded = model.encoder.batch_decode(x_final, attention_mask=attention_mask)

        for seq, dec, s, e, fid, v in zip(batch["sequence"], decoded, batch["cdr3_start"], batch["cdr3_end"],
                                          batch["File_ID"], batch["v_call"]):
            ok = len(dec) == len(seq)
            rows.append({
                "File_ID": fid, "v_call": v,
                "cdr3_true": seq[s:e],
                "cdr3_gen": dec[s:e] if ok else "",
                "framework_ok": bool(ok and dec[:s] == seq[:s] and dec[e:] == seq[e:]),
                "length_ok": ok,
            })
        print(f"  {len(rows)}/{len(ds)}", flush=True)

    df = pd.DataFrame(rows)
    df_ok = df[df["length_ok"]].copy()
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, f"{args.tag}.csv"), index=False)

    summary = summarize(df_ok, ref_h3, rng)
    summary["length_mismatches"] = int((~df["length_ok"]).sum())
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Baseline (approach A) summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nExamples (real -> generated):")
    for _, r in df_ok.head(10).iterrows():
        print(f"  {r['cdr3_true']} -> {r['cdr3_gen']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="results/inpaint_baseline")
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=None, help="default: generation.N_steps from the config (2000)")
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--selfcond_framework", action="store_true",
                        help="also feed the clean framework through the self-conditioning input")
    parser.add_argument("--adapter", type=str, default=None, help="path to a trained adapter checkpoint (last.pth)")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
