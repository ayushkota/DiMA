"""
Train the conditional adapter on OAS heavy chains. Everything else (ESM2, decoder, DiMA score estimator) is frozen.

Each training example:
  - clean latents of the full antibody are the target (as in normal DiMA training);
  - the context is the ESM2 encoding of the same antibody with the CDR-H3 letters replaced by <mask>;
  - all rows are noised at a random time t as in normal DiMA training;
  - the loss is the MSE between predicted and clean latents on the CDR-H3 rows only.

Example of usage:
python -m src.diffusion.train_adapter --config_path="src/configs" --offline --tag adapter_v1
"""
import argparse
import json
import math
import os
import time

import torch
from datasets import load_from_disk

from src.diffusion.inpaint_baseline import encode_masked, load_model
from src.models.adapter import ConditionAdapter, ConditionedScoreEstimator
from src.utils.hydra_utils import setup_config
from src.utils.training_utils import mse_loss


def compute_loss(model, wrapper, batch, max_sequence_len, t=None):
    device = model.device
    seqs = batch["sequence"]
    starts = torch.tensor(batch["cdr3_start"], device=device)
    ends = torch.tensor(batch["cdr3_end"], device=device)
    zeros = torch.zeros_like(starts)

    clean_X, mask, _ = encode_masked(model, seqs, zeros, zeros, max_sequence_len)  # nothing masked: the target
    context, _, h3 = encode_masked(model, seqs, starts, ends, max_sequence_len)    # CDR-H3 replaced by <mask>
    wrapper.set_context(context, mask, h3)

    B = clean_X.size(0)
    eps = 1e-5
    if t is None:
        t = torch.rand(B, device=device) * (model.dynamic.T - eps) + eps
    x_t = model.dynamic.marginal(clean_X, t)["x_t"]

    # self-conditioning exactly as in BaseDiffusionTrainer.calc_loss
    x_self = wrapper(x_t=x_t, time_t=t, attention_mask=mask, x_0_self_cond=torch.zeros_like(x_t))
    loss_self = mse_loss(clean_X, x_self, h3.float())
    x_0 = wrapper(x_t=x_t, time_t=t, attention_mask=mask, x_0_self_cond=x_self.detach())
    loss_x0 = mse_loss(clean_X, x_0, h3.float())
    return loss_x0 + loss_self, loss_x0, loss_self


@torch.no_grad()
def validate(model, wrapper, ds_val, args):
    wrapper.adapter.eval()
    losses = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(1234)  # same noise and times at every validation, so the numbers are comparable
        for i in range(args.val_batches):
            batch = ds_val[i * args.batch_size:(i + 1) * args.batch_size]
            _, loss_x0, _ = compute_loss(model, wrapper, batch, args.max_sequence_len)
            losses.append(loss_x0.item())
    wrapper.adapter.train()
    return sum(losses) / len(losses)


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
    for p in model.score_estimator.parameters():
        p.requires_grad_(False)

    adapter = ConditionAdapter(config.model.config, num_blocks=args.num_blocks).to(device)
    wrapper = ConditionedScoreEstimator(model.score_estimator, adapter).to(device)
    wrapper.score_estimator.eval()
    wrapper.adapter.train()
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter parameters: {n_params:,} (frozen score estimator: "
          f"{sum(p.numel() for p in model.score_estimator.parameters()):,})", flush=True)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    def lr_lambda(step):  # linear warmup, then cosine decay to 0
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    step = 0
    last_path = os.path.join(out_dir, "last.pth")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f"Resumed from step {step}", flush=True)

    def save(name):
        torch.save({"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "args": vars(args)},
                   os.path.join(out_dir, name))

    ds_train = load_from_disk(f"{args.data_dir}/train")
    ds_val = load_from_disk(f"{args.data_dir}/validation")
    log_file = open(os.path.join(out_dir, "log.jsonl"), "a")

    def log(record):
        print(json.dumps(record), flush=True)
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    if step == 0:
        # with the zero-initialised adapter this is the loss of the frozen pretrained model without any context
        log({"step": 0, "val_loss_x0_cdr_h3": validate(model, wrapper, ds_val, args)})

    t_start = time.time()
    start_step = step
    perm, pos = torch.randperm(len(ds_train), generator=torch.Generator().manual_seed(args.seed + step)), 0
    run_loss = run_n = 0
    while step < args.max_steps:
        optimizer.zero_grad()
        for _ in range(args.accum):
            if pos + args.batch_size > len(perm):
                perm, pos = torch.randperm(len(ds_train)), 0
            idx = perm[pos:pos + args.batch_size].tolist()
            pos += args.batch_size
            total, loss_x0, _ = compute_loss(model, wrapper, ds_train[idx], args.max_sequence_len)
            (total / args.accum).backward()
            run_loss += loss_x0.item(); run_n += 1
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            log({"step": step, "train_loss_x0_cdr_h3": run_loss / run_n, "lr": scheduler.get_last_lr()[0],
                 "sec_per_step": elapsed / (step - start_step),
                 "peak_mem_gb": [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]})
            run_loss = run_n = 0
        if step % args.val_every == 0 or step == args.max_steps:
            log({"step": step, "val_loss_x0_cdr_h3": validate(model, wrapper, ds_val, args)})
        if step % args.save_every == 0 or step == args.max_steps:
            save("last.pth")
        if (time.time() - t_start) / 60 > args.time_limit_minutes:
            print(f"Time limit reached at step {step}; saving and stopping (rerun with --resume).", flush=True)
            save("last.pth")
            break
    save("last.pth")
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--data_dir", type=str, default="data/oas")
    parser.add_argument("--out_dir", type=str, default="checkpoints/adapter")
    parser.add_argument("--tag", type=str, default="adapter_v1")
    parser.add_argument("--max_steps", type=int, default=2000, help="optimizer steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum", type=int, default=2, help="gradient accumulation: effective batch = batch_size * accum")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--max_sequence_len", type=int, default=160)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--time_limit_minutes", type=float, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
