"""Linear-probe evaluation of the teacher's frozen per-voxel features.

usage: python eval.py out/ckpt.pt --data_root=/path/to/pilarnet
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from data import PILArNet, collate_val, val_sample

LRS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1)
EXCLUDED = ((1, 1), (0, 2))  # (motif, pid) combinations that are label noise: track+electron, shower+muon


def probe_events(n_val=10_000):
    """A fixed random set of 250 train + 250 test events from the first 10k validation events."""
    return torch.randperm(n_val, generator=torch.Generator().manual_seed(0))[::4][:500].tolist()


@torch.no_grad()
def features(model, dataset, events, batch=6):
    xs, motifs, pids = [], [], []
    for i in range(0, len(events), batch):
        x = collate_val([val_sample(dataset[e], np.random.default_rng(e)) for e in events[i:i + batch]])
        feat = model.features({k: v.cuda() for k, v in x.items()}).float().cpu()
        for view in x["batch"].unique():
            keep = x["batch"] == view
            xs.append(feat[keep])
            motifs.append(x["segment_motif"][keep])
            pids.append(x["segment_pid"][keep])
    return xs, motifs, pids


def mean_f1(pred, y, classes):
    tp = torch.bincount(y[pred == y], minlength=classes).double()
    precision = tp / (torch.bincount(pred, minlength=classes) + 1e-10)
    recall = tp / (torch.bincount(y, minlength=classes) + 1e-10)
    return (2 * precision * recall / (precision + recall + 1e-10)).mean().item()


def linear_probe(x_train, y_train, x_test, y_test, epochs=10, batch=32768):
    """Fit one linear classifier per learning rate; report the best test mean F1."""
    classes = int(max(y_train.max(), y_test.max())) + 1
    heads = torch.nn.ModuleList(torch.nn.Linear(x_train.shape[1], classes) for _ in LRS).cuda()
    for h in heads:
        torch.nn.init.normal_(h.weight, std=0.01)
        torch.nn.init.zeros_(h.bias)
    opt = torch.optim.AdamW([dict(params=h.parameters(), lr=lr) for h, lr in zip(heads, LRS)], weight_decay=0.01)
    for _ in range(epochs):
        for idx in torch.randperm(len(x_train)).split(batch):
            x, y = x_train[idx].cuda(), y_train[idx].cuda()
            loss = sum(F.cross_entropy(h(x), y) for h in heads)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        preds = [torch.cat([h(x.cuda()).argmax(-1).cpu() for x in x_test.split(batch)]) for h in heads]
    return max(mean_f1(p, y_test, classes) for p in preds)


def evaluate(model, dataset):
    was_training = model.training
    model.eval()
    xs, motifs, pids = features(model, dataset, probe_events())
    x_train, x_test = torch.cat(xs[:250]), torch.cat(xs[250:])
    motif_train, motif_test = torch.cat(motifs[:250]), torch.cat(motifs[250:])
    results = {"val/mF1": linear_probe(x_train, motif_train, x_test, motif_test)}

    # joint motif x pid label: every observed combination is a class, minus the excluded ones
    joint = torch.stack([torch.cat(motifs), torch.cat(pids)], 1)
    keep = ~sum((joint == torch.tensor(c)).all(1) for c in EXCLUDED).bool()
    _, label = torch.unique(joint[keep], dim=0, return_inverse=True)
    split = len(motif_train)
    keep_train, keep_test = keep[:split], keep[split:]
    n_train = int(keep_train.sum())
    results["segment_motif+segment_pid/val/mF1"] = linear_probe(
        x_train[keep_train], label[:n_train], x_test[keep_test], label[n_train:])
    model.train(was_training)
    return results


if __name__ == "__main__":
    from model import Sonata
    args = dict(a[2:].split("=", 1) for a in sys.argv[2:])
    model = Sonata().cuda()
    model.load_state_dict(torch.load(sys.argv[1], map_location="cuda")["model"])
    dataset = PILArNet(args.get("data_root", os.environ.get("PILARNET_DATA_ROOT", "")), "val", max_len=10_000)
    for k, v in evaluate(model, dataset).items():
        print(f"{k}: {v:.4f}")
