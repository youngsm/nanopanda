"""LitePT backbone and Sonata self-distillation."""

from dataclasses import dataclass

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_qkvpacked_func

GRID_SIZE = 0.001
WINDOW = 1024  # attention window (points)
# the compiled attention stages need a handful of graphs (per stage; training, no-grad and eval)
torch._dynamo.config.recompile_limit = 16
# keep each compiled stage whole under DDP (splitting at gradient buckets fails with dynamic shapes)
torch._dynamo.config.optimize_ddp = False


def interleave(g, bits):
    """Bit b of axis d (counting from the top) goes to position 3 * (bits - b) - 1 - d."""
    b, d = (
        torch.arange(bits, device=g.device),
        torch.arange(3, device=g.device)[:, None],
    )
    return (((g[:, :, None] >> (bits - 1 - b)) & 1) << (3 * (bits - b) - 1 - d)).sum(
        (1, 2)
    )


@triton.jit(do_not_specialize=["n", "bits"])
def _hilbert_kernel(grid_ptr, code_ptr, n, bits, BLOCK: tl.constexpr):
    """Skilling's transpose algorithm, bit interleave and Gray decode, one point per lane."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    x = tl.load(grid_ptr + 3 * i, mask=valid, other=0)
    y = tl.load(grid_ptr + 3 * i + 1, mask=valid, other=0)
    z = tl.load(grid_ptr + 3 * i + 2, mask=valid, other=0)
    for b in range(bits):
        shift = (bits - 1 - b).to(tl.int64)
        low = (tl.full((), 1, tl.int64) << shift) - 1
        x = tl.where((x >> shift) & 1 == 1, x ^ low, x)
        for_y = (y >> shift) & 1 == 1
        x = tl.where(for_y, x ^ low, x)
        swap = tl.where(for_y, 0, (x ^ y) & low)
        x, y = x ^ swap, y ^ swap
        for_z = (z >> shift) & 1 == 1
        x = tl.where(for_z, x ^ low, x)
        swap = tl.where(for_z, 0, (x ^ z) & low)
        x, z = x ^ swap, z ^ swap
    code = tl.zeros_like(x)
    for b in range(bits):
        shift = (bits - 1 - b).to(tl.int64)
        top = 3 * (bits - b).to(tl.int64) - 1
        code |= ((x >> shift) & 1) << top
        code |= ((y >> shift) & 1) << (top - 1)
        code |= ((z >> shift) & 1) << (top - 2)
    for k in tl.static_range(6):  # Gray decode; shifts >= 3 * bits are no-ops
        code ^= code >> (1 << k)
    tl.store(code_ptr + i, code, mask=valid)


def hilbert(grid, bits):
    code = torch.empty(len(grid), dtype=torch.long, device=grid.device)
    _hilbert_kernel[(triton.cdiv(len(grid), 256),)](grid.contiguous(), code, len(grid), bits, BLOCK=256)
    return code


def serialize(grid, batch):
    """Four point orders (hilbert, hilbert-trans, z, z-trans) within each view, shuffled."""
    depth = int(grid.max() + 1).bit_length()
    both = torch.cat([grid, grid[:, [1, 0, 2]]])
    codes = torch.cat([hilbert(both, depth), interleave(both, depth)]).view(4, -1)
    codes = codes[torch.randperm(4)] | (batch << (3 * depth))
    order = codes.argsort(dim=1)
    return order, order.argsort(dim=1)


def windows(batch, size=WINDOW):
    """Attention windows over each view's serialized points: windows start every `size` points
    and the last one is shifted back to end at the view's last point. Returns the serialized positions of all windows concatenated, the window offsets, and
    each position's index in that concatenation (a point seen by two windows keeps its regular one)."""
    count = torch.bincount(batch)
    first = count.cumsum(0) - count  # each view's first serialized position
    width = count.clamp(max=size)
    per_view = (count + size - 1) // size
    first_window = per_view.cumsum(0) - per_view
    view = torch.repeat_interleave(per_view)  # the view of each window
    j = torch.arange(len(view), device=batch.device) - first_window[view]
    start = first[view] + torch.minimum(j * size, count[view] - width[view])
    offsets = F.pad(width[view].cumsum(0), (1, 0))
    index = torch.repeat_interleave(start - offsets[:-1], width[view])
    index += torch.arange(len(index), device=batch.device)
    t = torch.arange(len(batch), device=batch.device) - first[batch]
    window = first_window[batch] + torch.minimum(t // size, per_view[batch] - 1)
    where = offsets[window] + first[batch] + t - start[window]
    return index, offsets.int(), where


@dataclass
class Level:
    """One resolution of the point cloud. `parent`/`cluster` link a pooled level to the finer one."""

    feat: torch.Tensor
    coord: torch.Tensor
    origin: torch.Tensor
    grid: torch.Tensor
    batch: torch.Tensor
    parent: "Level" = None
    cluster: torch.Tensor = None  # which coarse point each parent point was pooled into
    sparse: spconv.SparseConvTensor = None  # voxel structure for convolutions

    def upcast(self, levels=99):
        """Walk back to finer levels, concatenating each coarse feature onto its children."""
        level = self
        while levels and level.parent is not None:
            level.parent.feat = torch.cat(
                [level.parent.feat, level.feat[level.cluster]], -1
            )
            level, levels = level.parent, levels - 1
        return level


def sparse_tensor(level):
    shape = (level.grid.max(0).values + 96).tolist()
    indices = torch.cat([level.batch[:, None], level.grid], 1).int()
    return spconv.SparseConvTensor(level.feat, indices, shape, int(level.batch[-1]) + 1)


def segment_max(x, segment, n):
    """Per-segment max whose gradient goes to exactly one winning point per channel (ties included);
    native scatter_reduce/segment_reduce spread and duplicate gradients over tied maxima."""
    with torch.no_grad():
        index = segment[:, None].expand_as(x)
        best = x.new_full((n, x.shape[1]), float("-inf")).scatter_reduce(
            0, index, x, "amax"
        )
        rows = torch.arange(len(x), device=x.device)[:, None].expand_as(x)
        rows = torch.where(x == best[segment], rows, len(x))
        winner = torch.full_like(best, len(x), dtype=torch.long).scatter_reduce(
            0, index, rows, "amin"
        )
    return x.gather(0, winner)


def segment_mean(x, segment, n):
    total = x.new_zeros(n, x.shape[1]).index_add_(0, segment, x)
    return total / torch.bincount(segment, minlength=n)[:, None]


def drop_path(x, p, training):
    """Stochastic depth, dropping individual points."""
    if p == 0 or not training:
        return x
    return x * x.new_empty(x.shape[0], 1).bernoulli_(1 - p) / (1 - p)


def rope_tables(pos, head_dim, base=100.0):
    """cos/sin for 3D rotary embeddings: each third of the head dim is rotated by one grid axis."""
    d = head_dim // 3
    angle = pos[:, None, :, None].float() * base ** (
        -torch.arange(0, d, 2, device=pos.device) / d
    )
    angle = torch.cat([angle, angle], -1)
    return angle.cos(), angle.sin()


def rope(x, cos, sin):
    x = x.float().unflatten(-1, (3, -1))
    x1, x2 = x.chunk(2, -1)
    return (x * cos + torch.cat([-x2, x1], -1) * sin).flatten(-2)


@torch.no_grad()
def plan_attention(level, head_dim):
    """Serialize a level once; each attention block of the stage uses one of its four orders.
    Per order: points -> window slots, window slots -> points, and the rope tables."""
    orders, inverses = serialize(level.grid, level.batch)
    index, offsets, where = windows(level.batch)
    plans = [
        (order[index], where[inverse], *rope_tables(level.grid[order[index]], head_dim))
        for order, inverse in zip(orders, inverses)
    ]
    return plans, offsets


class Attention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.qkv, self.proj = (
            nn.Linear(channels, 3 * channels),
            nn.Linear(channels, channels),
        )

    def forward(self, x, plan, offsets):
        gather, scatter, cos, sin = plan
        qkv = self.qkv(x)[gather].unflatten(1, (3, self.heads, -1))
        # attention runs in fp16; fp16 tensors stay out of indexing ops so the compiled backward works
        with torch.autocast("cuda", enabled=False):
            q, k, v = rope(qkv[:, 0], cos, sin), rope(qkv[:, 1], cos, sin), qkv[:, 2]
            qkv = torch.stack([q.half(), k.half(), v.half()], 1)
            out = flash_attn_varlen_qkvpacked_func(qkv, offsets, WINDOW)
        return self.proj(out.flatten(1).to(x.dtype)[scatter])


class AttnBlock(nn.Module):
    def __init__(self, channels, heads, drop, order_index):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(channels), Attention(channels, heads)
        self.norm2, self.mlp = (
            nn.LayerNorm(channels),
            nn.Sequential(
                nn.Linear(channels, 4 * channels),
                nn.GELU(),
                nn.Linear(4 * channels, channels),
            ),
        )
        self.drop, self.order_index = drop, order_index

    def forward(self, x, plans, offsets):
        attn = self.attn(self.norm1(x), plans[self.order_index], offsets)
        x = x + drop_path(attn, self.drop, self.training)
        return x + drop_path(self.mlp(self.norm2(x)), self.drop, self.training)


class AttnStage(nn.ModuleList):
    """The attention blocks of one stage, compiled as a single graph."""

    def forward(self, x, plans, offsets):
        for block in self:
            x = block(x, plans, offsets)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels, key):
        super().__init__()
        self.conv = spconv.SubMConv3d(channels, channels, 3, bias=True, indice_key=key)
        self.linear, self.norm = nn.Linear(channels, channels), nn.LayerNorm(channels)

    def forward(self, level):
        with torch.autocast("cuda", enabled=False):
            x = level.feat.float()
            y = self.conv(level.sparse.replace_feature(x)).features
            level.feat = (x + self.norm(self.linear(y))).to(level.feat.dtype)


class Pool(nn.Module):
    """Merge points sharing a voxel `stride` times larger: linear, max-pool, BatchNorm, GELU."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.proj, self.norm, self.stride = (
            nn.Linear(cin, cout),
            nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01),
            stride,
        )

    def forward(self, level):
        grid = torch.div(level.grid, self.stride, rounding_mode="trunc") | (
            level.batch[:, None] << 48
        )
        grid, cluster = torch.unique(grid, dim=0, return_inverse=True)
        n = len(grid)
        feat = F.gelu(self.norm(segment_max(self.proj(level.feat), cluster, n)))
        return Level(
            feat,
            segment_mean(level.coord, cluster, n),
            segment_mean(level.origin, cluster, n),
            grid & ((1 << 48) - 1),
            grid[:, 0] >> 48,
            parent=level,
            cluster=cluster,
        )


class LitePT(nn.Module):
    """Sparse-conv stages at fine resolution, windowed attention at coarse resolution."""

    def __init__(
        self,
        drop_path=0.3,
        channels=(54, 108, 432, 576),
        depths=(3, 6, 12, 6),
        heads=(3, 6, 24, 32),
        strides=(2, 1, 4),
        conv_stages=2,
        compile=True,
    ):
        super().__init__()
        self.stem = spconv.SubMConv3d(
            4, channels[0], 5, padding=1, bias=False, indice_key="stem"
        )
        self.stem_norm, self.conv_stages = nn.LayerNorm(channels[0]), conv_stages
        self.head_dims = [c // h for c, h in zip(channels, heads)]
        self.pools = nn.ModuleList(
            Pool(channels[s], channels[s + 1], strides[s]) for s in range(len(strides))
        )
        drops = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.blocks = nn.ModuleList(
            nn.ModuleList(ConvBlock(channels[s], f"stage{s}") for _ in range(depths[s]))
            if s < conv_stages
            else AttnStage(
                AttnBlock(channels[s], heads[s], drops[sum(depths[:s]) + i], i % 4)
                for i in range(depths[s])
            )
            for s in range(len(depths))
        )
        if compile:  # dynamic shapes: the number of points changes every batch
            for stage in self.blocks[conv_stages:]:
                stage.compile(dynamic=True)

    def forward(self, feat, coord, origin, batch, grid=None):
        if grid is None:
            grid = torch.div(
                coord - coord.min(0).values, GRID_SIZE, rounding_mode="trunc"
            )
        level = Level(feat, coord, origin, grid.long(), batch.long())
        level.sparse = sparse_tensor(level)
        level.feat = F.gelu(self.stem_norm(self.stem(level.sparse).features))
        for s, blocks in enumerate(self.blocks):
            if s > 0:
                level = self.pools[s - 1](level)
            if s < self.conv_stages:
                level.sparse = sparse_tensor(level)
                for block in blocks:
                    block(level)
            else:
                level.feat = blocks(level.feat, *plan_attention(level, self.head_dims[s]))
        return level


class Head(nn.Module):
    """Project to a unit embedding, score against 8192 unit prototypes (cosine similarities)."""

    def __init__(self, cin=1116, hidden=4096, embed=256, prototypes=8192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, embed)
        )
        for m in self.mlp[::2]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        self.prototypes = nn.Parameter(
            nn.Linear(embed, prototypes, bias=False).weight.detach()
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1) @ F.normalize(self.prototypes, dim=-1).T


def all_reduce(x):
    if dist.is_initialized():
        dist.all_reduce(x)
    return x


@torch.no_grad()
def sinkhorn(logits, temp, iters=3):
    """Balanced soft assignment of points to prototypes (jointly across ranks), one per loss term."""
    q = [(x.float() / temp).exp().T for x in logits]
    sizes, totals = all_reduce(
        torch.stack([torch.stack([x.new_tensor(x.shape[1]), x.sum()]) for x in q])
    ).T
    q = [x / t for x, t in zip(q, totals)]
    for _ in range(iters):
        rows = all_reduce(torch.stack([x.sum(1) for x in q]))
        q = [x / r[:, None] / len(r) for x, r in zip(q, rows)]
        q = [x / x.sum(0, keepdim=True) / n for x, n in zip(q, sizes)]
    return [(x * n).T for x, n in zip(q, sizes)]


@torch.no_grad()
def match(query, query_seg, ref, ref_seg, max_r=0.004):
    """Pairs (query index, nearest ref index) within the same segment id, closer than max_r."""
    query_order, ref_order = (
        query_seg.argsort(stable=True),
        ref_seg.argsort(stable=True),
    )
    segments = int(max(query_seg.max(), ref_seg.max())) + 1
    query_parts = query_order.split(
        torch.bincount(query_seg, minlength=segments).tolist()
    )
    ref_parts = ref_order.split(torch.bincount(ref_seg, minlength=segments).tolist())
    pairs, dists = [], []
    for qi, ri in zip(query_parts, ref_parts):
        for chunk in qi.split(8192) if len(ri) else []:
            d, j = torch.cdist(query[chunk].double(), ref[ri].double()).min(
                1
            )  # fp64: exact enough, no TF32
            pairs.append(torch.stack([chunk, ri[j]], 1))
            dists.append(d)
    return torch.cat(pairs)[torch.cat(dists) < max_r]


def per_view_mean(loss, view):
    """Mean over points within each view, then over the views that had any matches."""
    if len(loss) == 0:
        return loss.sum()
    total = loss.new_zeros(int(view.max()) + 1).index_add_(0, view, loss)
    count = torch.bincount(view)
    return (total[count > 0] / count[count > 0]).mean()


class Sonata(nn.Module):
    """Student sees masked global views and small local views; it predicts the EMA teacher's
    Sinkhorn-balanced prototype assignment at the matching points of the full global views."""

    def __init__(self, mask_size=0.04, mask_ratio=0.6, student_temp=0.1):
        super().__init__()
        self.student = nn.ModuleDict(
            dict(backbone=LitePT(drop_path=0.3), mask_head=Head(), unmask_head=Head())
        )
        self.teacher = nn.ModuleDict(
            dict(backbone=LitePT(drop_path=0.0), mask_head=Head(), unmask_head=Head())
        )
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.requires_grad_(False)
        self.mask_size, self.mask_ratio, self.student_temp = (
            mask_size,
            mask_ratio,
            student_temp,
        )

    @torch.no_grad()
    def mask(self, coord, view):
        """Drop a random 60% of the occupied 0.04-sized cells of each view."""
        lowest = coord.new_full((int(view.max()) + 1, 3), float("inf"))
        lowest = lowest.scatter_reduce(0, view[:, None].expand(-1, 3), coord, "amin")
        cell = torch.cat(
            [view[:, None], ((coord - lowest[view]) // self.mask_size).long()], 1
        )
        _, cell = torch.unique(cell, dim=0, return_inverse=True)
        n = int(cell.max()) + 1
        dropped = torch.zeros(n, dtype=torch.bool, device=coord.device)
        dropped[torch.randperm(n, device=coord.device)[: int(n * self.mask_ratio)]] = (
            True
        )
        return dropped[cell]

    def encode(self, net, x):
        return net.backbone(x["feat"], x["coord"], x["origin"], x["batch"]).upcast(2)

    def forward(self, batch, teacher_temp):
        glob, local = batch["glob"], batch["local"]
        with torch.no_grad():
            teacher = self.encode(self.teacher, glob)
            t_mask, t_unmask = (
                self.teacher.mask_head(teacher.feat),
                self.teacher.unmask_head(teacher.feat),
            )
        visible = ~self.mask(glob["coord"], glob["batch"])
        masked = self.encode(self.student, {k: v[visible] for k, v in glob.items()})
        s_mask = self.student.mask_head(masked.feat)
        views = self.encode(self.student, local)
        s_unmask = self.student.unmask_head(views.feat)

        # each student point is scored against the teacher at the nearest point of: its own global view
        # (mask), the event's other global view (roll_mask), or, for local views, the first global view
        principal = (teacher.batch % 2 == 0).nonzero()[:, 0]
        unmask = match(
            views.origin,
            views.batch // 6,
            teacher.origin[principal],
            teacher.batch[principal] // 2,
        )
        unmask[:, 1] = principal[unmask[:, 1]]
        terms = dict(  # student logits, teacher logits, (student, teacher) index pairs, student level
            mask=(
                s_mask,
                t_mask,
                match(masked.origin, masked.batch, teacher.origin, teacher.batch),
                masked,
            ),
            roll_mask=(
                s_mask,
                t_mask,
                match(masked.origin, masked.batch, teacher.origin, teacher.batch ^ 1),
                masked,
            ),
            unmask=(s_unmask, t_unmask, unmask, views),
        )
        targets = sinkhorn(
            [t[pairs[:, 1]] for _, t, pairs, _ in terms.values()], teacher_temp
        )

        out = {}
        for (key, (s, _, pairs, level)), target in zip(terms.items(), targets):
            ce = -(target * F.log_softmax(s[pairs[:, 0]] / self.student_temp, -1)).sum(
                -1
            )
            out[f"{key}_loss"] = per_view_mean(ce, level.batch[pairs[:, 0]])
        out["loss"] = (
            0.25 * out["mask_loss"]
            + 0.25 * out["roll_mask_loss"]
            + 0.5 * out["unmask_loss"]
        )
        return out

    @torch.no_grad()
    def update_teacher(self, momentum):
        student, teacher = (
            list(self.student.parameters()),
            list(self.teacher.parameters()),
        )
        torch._foreach_mul_(teacher, momentum)
        torch._foreach_add_(teacher, student, alpha=1 - momentum)

    @torch.no_grad()
    def features(self, x):
        """Teacher features at input resolution: all four levels concatenated (1170 channels)."""
        return (
            self.teacher.backbone(
                x["feat"], x["coord"], x["coord"], x["batch"], x["grid"]
            )
            .upcast()
            .feat
        )
