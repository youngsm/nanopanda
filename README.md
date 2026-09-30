# nanopanda

A minimal reproduction of Panda's self-supervised pretraining on PILArNet, without
the complexities of [pimm](https://github.com/DeepLearnPhysics/particle-imaging-models):
a LitePT backbone trained with Sonata self-distillation, plus a linear probe to
evaluate it.

| file       | what                                                                  |
|------------|-----------------------------------------------------------------------|
| `data.py`  | HDF5 reader, augmentations, multi-view crops, training stream         |
| `model.py` | LitePT backbone (sparse conv + windowed attention) and Sonata          |
| `train.py` | config, DDP, optimizer and schedules, EMA teacher, checkpoint/resume   |
| `eval.py`  | linear probe on frozen per-voxel features (mean F1)                   |

## Setup

```bash
uv sync                    # add --extra wandb for logging
```

Point `--data_root` (or `PILARNET_DATA_ROOT`) at a PILArNet directory with `train/`
and `val/` HDF5 shards.

## Train

```bash
# batch 24: 8 GPUs x 3 events, or 4 GPUs x 6
uv run torchrun --standalone --nproc_per_node=8 train.py --out_dir=out
uv run torchrun --standalone --nproc_per_node=4 train.py --out_dir=out --local_batch=6
```

The batch is `local_batch` x GPUs; there is no gradient accumulation. Every
`Config` field in `train.py` can be overridden with `--name=value`, e.g.
`--wandb_project=nanopanda`. Training resumes from `out_dir/ckpt.pt`.
The attention stages are compiled with `torch.compile`, so the first step takes
about a minute.

## Evaluate

```bash
uv run python eval.py out/ckpt.pt
```
