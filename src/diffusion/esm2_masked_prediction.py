"""
ESM2's own masked-letter prediction, as a reference for the DiMA infilling results.

Uses the same masks as inpaint_random (same seed, batch size and mask function) and, as a third mode, the CDR-H3 mask of
inpaint_baseline. ESM2 gets the masked sequence, and its language-model head gives a probability over the 20 amino
acids at every masked position. Two kinds of answer are scored:
  - top-1: the most likely letter at each position (one fixed answer);
  - sampled: letters drawn from those probabilities (4 independent draws, used to measure diversity).

Example of usage:
python -m src.diffusion.esm2_masked_prediction --config_path="src/configs" --offline
"""
import argparse
import json
import os

import numpy as np
import torch
from Bio.Align import substitution_matrices
from datasets import load_from_disk
from transformers import EsmForMaskedLM, EsmTokenizer

from src.diffusion.inpaint_baseline import AA, H3_END_MOTIFS, aa_freq, tv_distance
from src.diffusion.inpaint_random import make_mask
from src.utils.hydra_utils import setup_config

CLASSES = {**{a: "hydrophobic" for a in "AVILMC"}, **{a: "aromatic" for a in "FWY"}, **{a: "polar" for a in "STNQ"},
           **{a: "positive" for a in "KRH"}, **{a: "negative" for a in "DE"}, **{a: "special" for a in "GP"}}
N_DRAWS = 4


def letter_freq(letters):
    counts = np.array([(letters == a).sum() for a in AA], dtype=float)
    return counts / counts.sum()


def similarity(true, pred, rng):
    """Same-class rate and mean BLOSUM62 score, against the same predicted letters shuffled over the positions."""
    blosum = substitution_matrices.load("BLOSUM62")
    shuffled = pred.copy()
    rng.shuffle(shuffled)
    out = {}
    for tag, p in (("", pred), ("_chance", shuffled)):
        ok = np.array([a in CLASSES and b in CLASSES for a, b in zip(true, p)])
        out[f"same_class{tag}"] = float(np.mean([CLASSES[a] == CLASSES[b] for a, b in zip(true[ok], p[ok])]))
        out[f"blosum62_mean{tag}"] = float(np.mean([blosum[a][b] for a, b in zip(true[ok], p[ok])]))
    return out


@torch.no_grad()
def run_mode(mode, model, tokenizer, aa_ids, ds, freq, args, device):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    true, top1, draws, rows = [], [], [], []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start:start + args.batch_size]
        seqs = batch["sequence"]
        tok = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_sequence_len + 2, return_attention_mask=True)
        ids, n_tokens = tok["input_ids"], tok["input_ids"].size(1)
        if mode == "h3":
            pos = torch.arange(n_tokens)[None]
            starts, ends = torch.tensor(batch["cdr3_start"]), torch.tensor(batch["cdr3_end"])
            masked = (pos >= (starts + 1)[:, None]) & (pos < (ends + 1)[:, None])
        else:
            masked = make_mask([len(s) for s in seqs], args.mask_frac, mode, rng, n_tokens)
        masked_ids = ids.clone()
        masked_ids[masked] = tokenizer.mask_token_id
        logits = model(input_ids=masked_ids.to(device), attention_mask=tok["attention_mask"].to(device)).logits
        lm = logits.float().cpu()[masked][:, aa_ids]  # [n_masked, 20]
        probs = torch.softmax(lm, dim=-1)
        top1 += [AA[i] for i in lm.argmax(-1).tolist()]
        draws += [[AA[i] for i in d] for d in torch.multinomial(probs, N_DRAWS, replacement=True).tolist()]
        true += tokenizer.convert_ids_to_tokens(ids[masked].tolist())
        rows += (masked.nonzero()[:, 0] + start).tolist()
        print(f"  {mode}: {min(start + args.batch_size, len(ds))}/{len(ds)}", flush=True)

    true, top1, draws, rows = np.array(true), np.array(top1), np.array(draws), np.array(rows)
    pairs = [(i, j) for i in range(N_DRAWS) for j in range(i + 1, N_DRAWS)]
    per_seq = [float((top1[rows == r] == true[rows == r]).mean()) for r in np.unique(rows)]
    summary = {
        "mask_mode": mode,
        "masked_residues_total": int(len(true)),
        "top1_recovery": float((top1 == true).mean()),
        "top1_recovery_mean_over_sequences": float(np.mean(per_seq)),
        "sampled_recovery": float((draws[:, 0] == true).mean()),
        "sampled_diversity (fraction of positions where two independent draws differ)":
            float(np.mean([(draws[:, i] != draws[:, j]).mean() for i, j in pairs])),
        "composition_TV_top1_vs_real_masked": tv_distance(letter_freq(top1), letter_freq(true)),
        "chance_draw_from_composition": float((freq ** 2).sum()),
        "chance_always_most_common_letter": float(freq.max()),
        "top1_most_common_letter_share": float(max((top1 == a).mean() for a in AA)),
    }
    summary.update({f"top1_{k}": v for k, v in similarity(true, top1, rng).items()})
    if mode == "h3":
        gen = ["".join(top1[rows == r]) for r in np.unique(rows)]
        real = ["".join(true[rows == r]) for r in np.unique(rows)]
        summary["top1_starts_with_A"] = float(np.mean([g.startswith("A") for g in gen]))
        summary["top1_ends_with_J_motif"] = float(np.mean([g.endswith(H3_END_MOTIFS) for g in gen]))
        summary["real_starts_with_A"] = float(np.mean([r.startswith("A") for r in real]))
        summary["real_ends_with_J_motif"] = float(np.mean([r.endswith(H3_END_MOTIFS) for r in real]))
        summary["top1_unique_fraction"] = len(set(gen)) / len(gen)
    return summary


def main(args):
    config = setup_config(config_path=args.config_path)
    name = config.encoder.config.encoder_model_name
    if args.offline:
        from huggingface_hub import snapshot_download
        name = snapshot_download(name, local_files_only=True)
    device = torch.device("cuda:0")

    tokenizer = EsmTokenizer.from_pretrained(name)
    model = EsmForMaskedLM.from_pretrained(name, device_map="auto").eval()
    aa_ids = torch.tensor(tokenizer.convert_tokens_to_ids(list(AA)))

    ds_full = load_from_disk(f"{args.data_dir}/{args.split}")
    freq = aa_freq(ds_full["sequence"])
    ds = ds_full.select(range(min(args.n, len(ds_full))))

    report = {}
    for mode in args.modes:
        report[mode] = run_mode(mode, model, tokenizer, aa_ids, ds, freq, args, device)
        print(json.dumps(report[mode], indent=2), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "esm2_masked_prediction.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="results/inpaint_random")
    parser.add_argument("--modes", type=str, nargs="+", default=["scattered", "span", "h3"],
                        choices=["scattered", "span", "h3"])
    parser.add_argument("--mask_frac", type=float, default=0.15)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
