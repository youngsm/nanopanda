"""Sonata self-supervised pretraining of LitePT on PILArNet.

batch 24:  torchrun --nproc_per_node=8 train.py --data_root=... --out_dir=out
           torchrun --nproc_per_node=4 train.py --data_root=... --out_dir=out --local_batch=6
Each step is one forward/backward over local_batch x GPUs events. There is no gradient
accumulation: Sinkhorn and SyncBatchNorm must see the whole batch at once.
Any Config field can be overridden with --name=value.
"""
import ast
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data import PILArNet, TrainStream
from eval import evaluate
from model import Sonata


@dataclass
class Config:
    data_root: str = os.environ.get("PILARNET_DATA_ROOT", "")  # contains train/ and val/ shards
    out_dir: str = "out"
    subset_size: int = 1_000_000  # unique training events
    steps: int = 416_666          # ~10M event presentations at batch 24
    local_batch: int = 3          # events per GPU; the batch is local_batch x GPUs (24 = 8 x 3)
    lr: float = 4.2e-3            # transformer/conv blocks train at lr / 10
    weight_decay: float = 0.05
    warmup: float = 0.05          # fraction of steps for LR and teacher-temperature warmup
    clip_grad: float = 3.0
    teacher_temp: tuple = (0.04, 0.07)
    momentum: tuple = (0.996, 1.0)  # teacher EMA, cosine to 1
    eval_every: int = 10_000
    save_every: int = 5_000
    log_every: int = 10
    workers: int = 8
    seed: int = 0
    wandb_project: str = ""       # empty: no wandb


def parse_value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    cfg = Config(**{k: parse_value(v) for k, v in (a[2:].split("=", 1) for a in sys.argv[1:])})
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    torch.manual_seed(cfg.seed + rank)
    torch.set_float32_matmul_precision("high")

    model = Sonata().cuda()
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 4 parameter groups: {blocks, everything else} x {weight decay, none for biases/norms}
    groups = []
    for is_block in (False, True):
        for decay in (True, False):
            params = [p for n, p in model.student.named_parameters()
                      if (".blocks." in n) == is_block and (p.ndim > 1) == decay]
            groups.append(dict(params=params, lr=cfg.lr / (10 if is_block else 1),
                               weight_decay=cfg.weight_decay if decay else 0.0))
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=cfg.steps, pct_start=cfg.warmup,
        div_factor=10, final_div_factor=1000, base_momentum=0.85, max_momentum=0.95)

    # attention runs in fp16, whose tiny backward gradients underflow without loss scaling
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start = ckpt["step"]
    net = DDP(model, device_ids=[torch.cuda.current_device()]) if ddp else model

    stream = TrainStream(PILArNet(cfg.data_root, "train"), cfg.subset_size, cfg.seed,
                         cfg.local_batch, rank, world, start)
    batches = iter(DataLoader(stream, batch_size=None, num_workers=cfg.workers, pin_memory=True))
    val = PILArNet(cfg.data_root, "val", max_len=10_000)
    if rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)
        if cfg.wandb_project:
            import wandb
            run = os.path.basename(os.path.abspath(cfg.out_dir))  # resumes append to the same run
            wandb.init(project=cfg.wandb_project, id=run, name=run, config=asdict(cfg), resume="allow")

    warmup = int(cfg.warmup * cfg.steps)
    t0 = time.time()
    for step in range(start, cfg.steps):
        lo, hi = cfg.teacher_temp
        teacher_temp = lo + (hi - lo) * step / (warmup - 1) if step < warmup else hi
        m0, m1 = cfg.momentum
        momentum = m1 + 0.5 * (m0 - m1) * (1 + math.cos(math.pi * step / cfg.steps))

        batch = {view: {k: v.cuda(non_blocking=True) for k, v in x.items()} for view, x in next(batches).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(batch, teacher_temp)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg.clip_grad)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        model.update_teacher(momentum)

        if step % cfg.log_every == 0:
            logs = {k: v.detach() for k, v in out.items()}
            if ddp:
                for v in logs.values():
                    dist.all_reduce(v, op=dist.ReduceOp.AVG)
            logs = {f"train/{k}": v.item() for k, v in logs.items()}
            logs.update({"train/grad_norm": grad_norm.item(), "train/lr": sched.get_last_lr()[0],
                         "train/teacher_temp": teacher_temp, "train/momentum": momentum,
                         "train/loss_scale": scaler.get_scale()})
            if rank == 0:
                dt, t0 = (time.time() - t0) / (cfg.log_every if step > start else 1), time.time()
                print(f"step {step} loss {logs['train/loss']:.4f} grad {logs['train/grad_norm']:.2f} {dt:.2f}s/it",
                      flush=True)
                if cfg.wandb_project:
                    wandb.log(logs, step=step)
        if (step + 1) % cfg.eval_every == 0 and rank == 0:
            results = evaluate(model, val)
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in results.items()), flush=True)
            if cfg.wandb_project:
                wandb.log(results, step=step)
        if (step + 1) % cfg.save_every == 0 and rank == 0:
            state = dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         scheduler=sched.state_dict(), scaler=scaler.state_dict(), step=step + 1,
                         config=asdict(cfg))
            torch.save(state, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        if ddp and ((step + 1) % cfg.eval_every == 0 or (step + 1) % cfg.save_every == 0):
            dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
