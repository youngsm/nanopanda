"""PILArNet events -> augmented multi-view training batches."""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

os.environ.setdefault(
    "HDF5_USE_FILE_LOCKING", "FALSE"
)  # some shared filesystems reject HDF5 file locks
import h5py  # noqa: E402

GRID_SIZE = 0.001  # voxel size in normalized coordinates
N_GLOBAL, N_LOCAL = 2, 6


class PILArNet(Dataset):
    """Events with >= min_points raw hits, read straight from the PILArNet HDF5 shards."""

    def __init__(self, root, split, min_points=1024, energy_threshold=0.13, max_len=-1):
        self.files = sorted(glob.glob(os.path.join(root, f"*{split}/*.h5")))
        assert self.files, f"no {split} shards under {root!r}: set PILARNET_DATA_ROOT (see README)"
        self.rows = [
            np.flatnonzero(np.load(f.replace(".h5", "_points.npy")) >= min_points)
            for f in self.files
        ]
        self.ends = np.cumsum([len(r) for r in self.rows])
        self.length = (
            int(self.ends[-1]) if max_len < 0 else min(max_len, int(self.ends[-1]))
        )
        self.energy_threshold, self.h5 = energy_threshold, None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if (
            self.h5 is None
        ):  # open lazily so every dataloader worker gets its own handles
            self.h5 = [h5py.File(f, "r") for f in self.files]
        f = int(np.searchsorted(self.ends, i, side="right"))
        row = self.rows[f][i - (self.ends[f - 1] if f else 0)]
        hits = self.h5[f]["point"][row].reshape(
            -1, 8
        )  # x, y, z voxel index, energy (MeV), ...
        clusters = self.h5[f]["cluster"][row].reshape(
            -1, 6
        )  # size, id, group, interaction, motif, pid
        motif, pid = (np.repeat(clusters[:, c], clusters[:, 0]) for c in (4, 5))
        pid[pid == -1] = 5  # LED deposits have no particle
        keep = hits[:, 3] > self.energy_threshold
        return dict(
            coord=hits[keep, :3],
            energy=hits[keep, 3:4],
            segment_motif=motif[keep],
            segment_pid=pid[keep],
        )


# ---- augmentations: numpy functions on a dict of per-point arrays ----


def normalize(d):
    d["coord"] = (
        d["coord"] - 384.0
    ) / 665.1075101064489  # detector center / half-diagonal (768 * sqrt(3) / 2)
    return d


def fnv_hash(grid):
    key = np.full(len(grid), 14695981039346656037, dtype=np.uint64)
    for j in range(3):
        key = (key * np.uint64(1099511628211)) ^ grid[:, j].astype(np.uint64)
    return key


def grid_sample(d, rng):
    """Keep one random point per voxel; its energy becomes the voxel's summed energy."""
    grid = np.floor(d["coord"] / GRID_SIZE).astype(np.int64)
    grid -= grid.min(0)
    key = fnv_hash(grid)
    order = np.argsort(key)
    _, voxel_sorted, count = np.unique(
        key[order], return_inverse=True, return_counts=True
    )
    pick = order[
        np.cumsum(count) - count + rng.integers(0, count.max(), count.size) % count
    ]
    voxel = np.empty_like(voxel_sorted)
    voxel[order] = voxel_sorted
    energy = np.zeros((len(count), 1), dtype=d["energy"].dtype)
    np.add.at(energy, voxel, d["energy"])
    d = {k: v[pick] for k, v in d.items()}
    d["energy"], d["grid"] = energy, grid[pick]
    return d


def log_energy(e, lo=0.01, hi=20.0):
    """Map [0, hi] MeV to about [-1, 1] on a log scale."""
    return (
        2 * (np.log10(e + lo) - np.log10(lo)) / (np.log10(hi + lo) - np.log10(lo)) - 1
    )


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return {
        "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }[axis]


def augment_view(v, rng, jitter):
    """Per-view geometry: center, scale, rotate, flip, log energy, jitter (origin stays untouched)."""
    coord = v["coord"] - (v["coord"].min(0) + v["coord"].max(0)) / 2
    coord = coord * rng.uniform(0.9, 1.1)
    for axis in "zxy":
        if rng.random() < 0.8:
            coord = coord @ np.transpose(rotation(axis, rng.uniform(-1, 1) * np.pi))
    for axis in range(3):
        if rng.random() < 0.5:
            coord[:, axis] = -coord[:, axis]
    v["coord"] = coord + np.clip(
        jitter * rng.standard_normal(coord.shape), -0.001, 0.001
    )
    v["energy"] = log_energy(v["energy"])
    return v


def make_views(d, rng, max_size=30000):
    """2 global + 6 local crops (k nearest points to a center). Only the global views see energy jitter;
    local centers are drawn from the part of the main global view not yet covered by a local view."""
    n = min(max_size, len(d["coord"]))

    def crop(src, center, scale):
        size = max(1, min(n, int(rng.uniform(*scale) * n)))
        index = np.argsort(np.square(d["coord"] - center).sum(1))[:size]
        return index, {k: src[k][index] for k in ("coord", "origin", "energy")}

    jittered = dict(d)
    if rng.random() < 0.8:
        jittered["energy"] = d["energy"] * (
            1 + np.clip(0.05 * rng.standard_normal(d["energy"].shape), -0.1, 0.1)
        )
    main_index, main = crop(
        jittered, d["coord"][rng.integers(len(d["coord"]))], (0.4, 1.0)
    )
    globals_ = [main] + [
        crop(jittered, main["coord"][rng.integers(len(main_index))], (0.4, 1.0))[1]
        for _ in range(N_GLOBAL - 1)
    ]
    covered, locals_ = np.zeros(len(main_index), dtype=bool), []
    for _ in range(N_LOCAL):
        if covered.all():
            covered[:] = False
        index, view = crop(
            d, main["coord"][rng.choice(np.flatnonzero(~covered))], (0.1, 0.4)
        )
        covered |= np.isin(main_index, index)
        locals_.append(view)
    return (
        [augment_view(v, rng, 1.25e-4) for v in globals_],
        [augment_view(v, rng, 2.5e-4) for v in locals_],
    )


def train_sample(event, rng):
    d = normalize(dict(event))
    d["coord"] = d["coord"] * rng.uniform(0.9, 1.2)
    d = grid_sample(d, rng)
    d["origin"] = d["coord"].copy()  # common frame for matching points across views
    return make_views(d, rng)


def val_sample(event, rng):
    d = grid_sample(normalize(dict(event)), rng)
    d["energy"] = log_energy(d["energy"])
    return d


# ---- batching ----


def stack_views(views):
    """Concatenate views into one point set; `batch` is the view index of each point."""
    cat = lambda k: torch.from_numpy(
        np.concatenate([v[k] for v in views]).astype(np.float32)
    )
    coord, energy = cat("coord"), cat("energy")
    batch = torch.cat([torch.full((len(v["coord"]),), i) for i, v in enumerate(views)])
    return dict(
        feat=torch.cat([coord, energy], 1),
        coord=coord,
        origin=cat("origin"),
        batch=batch,
    )


def collate_train(samples):
    return dict(
        glob=stack_views([v for g, _ in samples for v in g]),
        local=stack_views([v for _, l in samples for v in l]),
    )


def collate_val(events):
    out = stack_views([dict(e, origin=e["coord"]) for e in events])
    out["grid"] = torch.from_numpy(np.concatenate([e["grid"] for e in events]))
    for k in ("segment_motif", "segment_pid"):
        out[k] = torch.from_numpy(
            np.concatenate([e[k] for e in events]).astype(np.int64)
        )
    return out


class TrainStream(IterableDataset):
    """Yields this rank's batch for each step. Step s consumes event ordinals [s*B, (s+1)*B) with
    B = local_batch x world; ordinals walk a fresh permutation of the training subset every pass, and
    each sample's augmentation RNG is seeded by its ordinal, so resuming only needs the step number."""

    def __init__(
        self, dataset, subset_size, seed, local_batch, rank, world, start_step
    ):
        self.dataset, self.seed, self.local_batch = dataset, seed, local_batch
        self.rank, self.start_step, self.global_batch = (
            rank,
            start_step,
            local_batch * world,
        )
        self.subset = np.random.default_rng(0).permutation(len(dataset))[
            :subset_size
        ]  # fixed 1M-event pool
        self.cycle, self.perm = None, None

    def event(self, ordinal):
        cycle, i = divmod(ordinal, len(self.subset))
        if cycle != self.cycle:
            generator = torch.Generator().manual_seed(self.seed + cycle)
            self.cycle, self.perm = (
                cycle,
                torch.randperm(len(self.subset), generator=generator),
            )
        return self.dataset[int(self.subset[self.perm[i]])]

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        step = (
            self.start_step + worker
        )  # workers take turns; the DataLoader reads them round-robin
        while True:
            first = step * self.global_batch + self.rank * self.local_batch
            yield collate_train(
                [
                    train_sample(self.event(o), np.random.default_rng([self.seed, o]))
                    for o in range(first, first + self.local_batch)
                ]
            )
            step += workers
