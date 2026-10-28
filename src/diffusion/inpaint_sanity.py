"""
Sanity check for the inpainting loop in inpaint_baseline.py.

Runs the same loop, but at a few steps decodes the model's OWN prediction of the clean rows (x_0) and measures
how well the framework positions are recovered. If the framework rows are fed to the model at the right noise
level, this accuracy should be high and should rise as the noise level t falls.
The accuracy on the CDR-H3 positions is reported next to it for comparison.

Example of usage:
python -m src.diffusion.inpaint_sanity --config_path="src/configs" --n 100 --offline
"""
import argparse

import numpy as np
import torch
from datasets import load_from_disk

from src.diffusion.inpaint_baseline import encode_masked, load_model
from src.utils.hydra_utils import setup_config


def accuracy(decoded, seqs, starts, ends):
    fw_ok = fw_n = h3_ok = h3_n = 0
    for dec, seq, s, e in zip(decoded, seqs, starts, ends):
        if len(dec) != len(seq):
            fw_n += len(seq) - (e - s)
            h3_n += e - s
            continue
        match = np.array([a == b for a, b in zip(dec, seq)])
        fw = np.ones(len(seq), dtype=bool)
        fw[s:e] = False
        fw_ok += match[fw].sum(); fw_n += fw.sum()
        h3_ok += match[s:e].sum(); h3_n += e - s
    return fw_ok / fw_n, h3_ok / h3_n


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
    model = load_model(config, device)
    dyn = model.dynamic
    n_steps, t_min = config.generation.N_steps, config.generation.t_min
    ts = torch.linspace(dyn.T, t_min, n_steps + 1, device=device)
    report_at = {int(f * (n_steps - 1)) for f in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}

    ds = load_from_disk(f"{args.data_dir}/{args.split}").select(range(args.n))
    results = {i: [] for i in sorted(report_at)}

    for start in range(0, len(ds), args.batch_size):
        batch = ds[start:start + args.batch_size]
        starts = torch.tensor(batch["cdr3_start"], device=device)
        ends = torch.tensor(batch["cdr3_end"], device=device)
        c, mask, h3 = encode_masked(model, batch["sequence"], starts, ends, args.max_sequence_len)
        framework = (mask.bool() & ~h3)[..., None]
        B = c.size(0)

        x = torch.randn_like(c)
        x_self = torch.zeros_like(c)
        for i in range(n_steps):
            t = ts[i] * torch.ones(B, device=device)
            next_t = ts[i + 1] * torch.ones(B, device=device)
            out = model.solver.step(x_t=x, t=t, next_t=next_t, mask=mask, x_0_self_cond=x_self)
            x, x_self = out["x"], out["x_0"]
            if i in report_at:
                dec = model.encoder.batch_decode(x_self, attention_mask=mask)
                results[i].append(accuracy(dec, batch["sequence"], batch["cdr3_start"], batch["cdr3_end"]))
            p = dyn.marginal_params(next_t)
            x = torch.where(framework, p["mu"] * c + p["std"] * torch.randn_like(c), x)
        print(f"  {start + B}/{len(ds)}", flush=True)

    print("\n=== Accuracy of the model's own clean-row prediction (decoded), averaged over batches ===")
    print("step   t       mu(t)  framework_acc  cdr_h3_acc")
    for i in sorted(results):
        fw = float(np.mean([r[0] for r in results[i]]))
        h3 = float(np.mean([r[1] for r in results[i]]))
        mu = float(dyn.marginal_params(ts[i + 1].view(1))["mu"])
        print(f"{i:<6d} {float(ts[i + 1]):.3f}   {mu:.3f}  {fw:.4f}         {h3:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
