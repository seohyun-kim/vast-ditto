#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-2 inference:
- 최신 stage2 체크포인트를 로드하여 (6,512,256) 임베딩 윈도우 입력 → 원시(raw) 트레이스 6개 특성(cluster_id, cluster_center, offset, opcode, length, timediff) 예측
- 원본(파란색) vs 생성(주황색)을 6개 subplot으로 시각화(시간축=512)
- 차이 보고: scalar → MAE/MSE, categorical → Accuracy


python infer_autoencoder_stage2.py \
  --embed-dir out/emb_1 \
  --csv-dir   out/emb_1/cluster \
  --spec      out/emb_1/rowemb_spec.json \
  --params-npz out/emb_1/rowemb_params.npz \
  --ckpt-dir  ./outputs_stage2 \
  --batch-size 4 \
  --plot-out ./outputs_stage2/infer_win0.png \
  --metrics-out ./outputs_stage2/infer_metrics.npy

  
  
"""

from __future__ import annotations
import os, glob, json, argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# 공통 상수/유틸
# ----------------------------
FEATURES = ["cluster_id", "cluster_center", "offset", "opcode", "length", "timediff"]
SCALAR_FEATS = ["cluster_center", "offset", "length", "timediff"]
CATEG_FEATS  = ["opcode", "cluster_id"]
IGNORE_INDEX = -100

def list_npy_files(embed_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy found in {embed_dir}")
    return files

# ----------------------------
# Spec / Params (stage-2 학습 스펙/파라미터 구조와 동일)
# ----------------------------
class SpecBundle:
    def __init__(self, opcode_vocab, cluster_id_min, cluster_id_max, emb_dim, meta):
        self.opcode_vocab = opcode_vocab
        self.cluster_id_min = cluster_id_min
        self.cluster_id_max = cluster_id_max
        self.emb_dim = emb_dim
        self.meta = meta

def load_spec(spec_path: str) -> SpecBundle:
    with open(spec_path, "r") as f:
        s = json.load(f)
    return SpecBundle(
        opcode_vocab=list(s.get("opcode_vocab", [])),
        cluster_id_min=int(s.get("cluster_id_min", 0)),
        cluster_id_max=int(s.get("cluster_id_max", 0)),
        emb_dim=int(s.get("emb_dim", 256)),
        meta=dict(s.get("meta", {})),
    )

class ParamBundle:
    def __init__(self, W_opcode, W_cluster_id, scalars):
        self.W_opcode = W_opcode
        self.W_cluster_id = W_cluster_id
        self.scalars = scalars  # dict[feat] -> dict of arrays

def load_params_npz(path: str) -> ParamBundle:
    arrs = np.load(path)
    W_op   = arrs.get("categorical/opcode.weight", None)
    W_cid  = arrs.get("categorical/cluster_id.weight", None)
    scalars: Dict[str, Dict[str, np.ndarray]] = {}
    for feat in SCALAR_FEATS:
        d: Dict[str, np.ndarray] = {}
        for k in ("e_min","e_max","table","bin_edges"):
            if f"scalar/{feat}.{k}" in arrs.files:
                d[k] = arrs[f"scalar/{feat}.{k}"]
        scalars[feat] = d
    return ParamBundle(W_op, W_cid, scalars)

def validate_spec_params(spec: SpecBundle, params: ParamBundle):
    assert spec.emb_dim == 256, f"emb_dim must be 256, got {spec.emb_dim}"
    if params.W_opcode is not None:
        r, d = params.W_opcode.shape
        assert d == spec.emb_dim
        assert r == len(spec.opcode_vocab)
    if params.W_cluster_id is not None:
        r, d = params.W_cluster_id.shape
        assert d == spec.emb_dim
        expected = (spec.cluster_id_max - spec.cluster_id_min + 1) + 1
        assert r == expected, f"cluster_id rows {r} != expected {expected}"
    for feat in SCALAR_FEATS:
        sc = params.scalars.get(feat, {})
        if "e_min" in sc and "e_max" in sc:
            assert sc["e_min"].shape == (spec.emb_dim,)
            assert sc["e_max"].shape == (spec.emb_dim,)

# ----------------------------
# 데이터셋 (임베딩 + CSV 라벨, stage-2 학습 로더를 간소화)
# ----------------------------
def build_op2id(vocab: List[str]) -> Dict[str, int]:
    return {op: i for i, op in enumerate(vocab)}

def map_opcode(op: str, op2id: Dict[str,int]) -> int:
    return op2id.get(op, IGNORE_INDEX)

def map_cluster_id(cid_val: int, cid_min: int, cid_max: int) -> int:
    if cid_min <= cid_val <= cid_max:
        return (cid_val - cid_min) + 1
    return IGNORE_INDEX

class _CSVLabelCache:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.mem: Dict[str, Dict[str, np.ndarray]] = {}

    def _cache_path(self, csv_path: str) -> str:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        return os.path.join(self.cache_dir, base + ".npz") if self.cache_dir else ""

    def load_or_build(self, csv_path: str, op2id: Dict[str,int], cid_min: int, cid_max: int) -> Dict[str, np.ndarray]:
        if csv_path in self.mem:
            return self.mem[csv_path]
        if self.cache_dir:
            cp = self._cache_path(csv_path)
            if os.path.isfile(cp):
                arrs = np.load(cp)
                obj = {k: arrs[k] for k in arrs.files}
                self.mem[csv_path] = obj
                return obj

        op_ids, cids = [], []
        cc, off, leng, td = [], [], [], []
        import csv as _csv
        with open(csv_path, "r", newline="") as f:
            r = _csv.DictReader(f)
            for row in r:
                op = (row.get("opcode", "") or "").strip()
                try:
                    cid_val = int(row.get("cluster_id", "0") or 0)
                except ValueError:
                    cid_val = cid_min - 1
                op_ids.append(map_opcode(op, op2id))
                cids.append(map_cluster_id(cid_val, cid_min, cid_max))
                cc.append(float(row.get("cluster_center", "0") or 0.0))
                off.append(float(row.get("offset", "0") or 0.0))
                leng.append(float(row.get("length", "0") or 0.0))
                td.append(float(row.get("timediff", "0") or 0.0))

        obj = {
            "opcode":         np.asarray(op_ids, dtype=np.int64),
            "cluster_id":     np.asarray(cids,   dtype=np.int64),
            "cluster_center": np.asarray(cc,     dtype=np.float32),
            "offset":         np.asarray(off,    dtype=np.float32),
            "length":         np.asarray(leng,   dtype=np.float32),
            "timediff":       np.asarray(td,     dtype=np.float32),
        }
        if self.cache_dir:
            np.savez_compressed(self._cache_path(csv_path), **obj)
        self.mem[csv_path] = obj
        return obj

class EmbeddingCSVWindowDataset(Dataset):
    """
    *.npy: (N,6,256) → 슬라이딩 윈도우 (6,512,256)
    *.csv: 같은 범위 [s:e) 라벨(y_cat: opcode/cluster_id, y_raw: 4개 스칼라)
    """
    def __init__(self, embed_dir: str, csv_dir: str, spec: SpecBundle,
                 win: int = 512, stride: int = 256, dtype=torch.float32,
                 label_cache_dir: Optional[str] = "out/label_cache"):
        super().__init__()
        self.embed_dir = embed_dir
        self.csv_dir   = csv_dir
        self.spec      = spec
        self.win       = int(win)
        self.stride    = int(stride)
        self.dtype     = dtype

        self.op2id   = build_op2id(list(spec.opcode_vocab))
        self.cid_min = int(spec.cluster_id_min)
        self.cid_max = int(spec.cluster_id_max)

        self.embed_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))
        if not self.embed_files:
            raise FileNotFoundError(f"No .npy in {embed_dir}")
        self.csv_files   = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        if not self.csv_files:
            raise FileNotFoundError(f"No .csv in {csv_dir}")

        self.file_meta: List[Dict[str, Any]] = []
        for npy in self.embed_files:
            stem = os.path.splitext(os.path.basename(npy))[0]
            cand = [p for p in self.csv_files if os.path.splitext(os.path.basename(p))[0] == stem]
            if not cand:
                cand = [p for p in self.csv_files if stem in os.path.basename(p)]
            if not cand and len(self.csv_files) == 1:
                cand = [self.csv_files[0]]
            if not cand:
                raise FileNotFoundError(f"CSV matching '{stem}' not found in {csv_dir}")
            csv_path = cand[0]
            arr = np.load(npy, mmap_mode="r")
            if arr.ndim != 3 or arr.shape[1] != 6 or arr.shape[2] != 256:
                raise ValueError(f"Bad shape {arr.shape} in {npy}, expected (N,6,256)")
            N = arr.shape[0]
            self.file_meta.append({"npy": npy, "csv": csv_path, "N": N})

        self._label_cache = _CSVLabelCache(label_cache_dir)
        for meta in self.file_meta:
            _ = self._label_cache.load_or_build(meta["csv"], self.op2id, self.cid_min, self.cid_max)

        self._emb_mm: Dict[str, np.memmap] = {}
        self.index: List[Tuple[int,int]] = []
        for fi, meta in enumerate(self.file_meta):
            N = meta["N"]
            if N < self.win: 
                continue
            for s in range(0, N - self.win + 1, self.stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"No windows with win={self.win}, stride={self.stride}")

    def __len__(self): return len(self.index)

    def _emb_arr(self, npy_path: str) -> np.memmap:
        arr = self._emb_mm.get(npy_path)
        if arr is None:
            arr = np.load(npy_path, mmap_mode="r")
            self._emb_mm[npy_path] = arr
        return arr

    def __getitem__(self, idx: int):
        fi, s = self.index[idx]
        e = s + self.win
        meta = self.file_meta[fi]

        arr = self._emb_arr(meta["npy"])         # (N,6,256)
        win_np = arr[s:e]                         # (512,6,256)
        x = torch.from_numpy(win_np.transpose(1,0,2).copy()).to(self.dtype)  # (6,512,256)

        lbl_np = self._label_cache.mem[meta["csv"]]
        y_cat = {
            "opcode":     torch.from_numpy(lbl_np["opcode"][s:e]).long(),      # (512,)
            "cluster_id": torch.from_numpy(lbl_np["cluster_id"][s:e]).long(),  # (512,)
        }
        y_raw = {
            "cluster_center": torch.from_numpy(lbl_np["cluster_center"][s:e]).float(),
            "offset":         torch.from_numpy(lbl_np["offset"][s:e]).float(),
            "length":         torch.from_numpy(lbl_np["length"][s:e]).float(),
            "timediff":       torch.from_numpy(lbl_np["timediff"][s:e]).float(),
        }
        return x, y_cat, y_raw

def collate_fn(batch):
    xs, ycats, yraws = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,6,512,256)
    def stack_long(key):  return torch.stack([yc[key] for yc in ycats], dim=0).long()
    def stack_float(key): return torch.stack([yr[key] for yr in yraws],  dim=0).float()
    y_cat = {"opcode": stack_long("opcode"), "cluster_id": stack_long("cluster_id")}
    y_raw = {f: stack_float(f) for f in SCALAR_FEATS}
    return x, y_cat, y_raw

# ----------------------------
# 모델 (stage-2 학습과 동일 구조)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        groups = 8
        if c_out % groups != 0:
            for g in (4,2,1):
                if c_out % g == 0:
                    groups = g; break
        self.norm = nn.GroupNorm(groups, c_out)
        self.act  = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.b1 = ConvBlock(c_in, c_out)
        self.b2 = ConvBlock(c_out, c_out)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        h = self.b2(self.b1(x))
        y = self.down(h)
        return y, h

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.b1 = ConvBlock(c_out + c_skip, c_out)
        self.b2 = ConvBlock(c_out, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(f"Up mismatch: {x.shape} vs {skip.shape}")
        x = torch.cat([x, skip], dim=1)
        x = self.b1(x); x = self.b2(x)
        return x

class UpNoSkip(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.b1 = ConvBlock(c_out, c_out)
        self.b2 = ConvBlock(c_out, c_out)
    def forward(self, x):
        x = self.up(x); x = self.b1(x); x = self.b2(x)
        return x

class AEBackbone(nn.Module):
    def __init__(self, c_in=6, c_lat=8):
        super().__init__()
        self.down1 = Down(c_in, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, c_lat)
        self.mid   = ConvBlock(c_lat, c_lat)
        self.up3   = Up(c_in=c_lat, c_skip=128, c_out=128)
        self.up2   = Up(c_in=128,   c_skip=64,  c_out=64)
        self.up1   = UpNoSkip(c_in=64, c_out=64)
        self.final = nn.Conv2d(64, c_in, kernel_size=1)
    def encode(self, x):
        x1, _ = self.down1(x)
        x2, _ = self.down2(x1)
        x3, _ = self.down3(x2)
        z = self.mid(x3)
        return z, (x2, x1)
    def decode(self, z, skips):
        x2, x1 = skips
        u2 = self.up3(z, x2)
        u1 = self.up2(u2, x1)
        u0 = self.up1(u1)
        y  = self.final(u0)
        return y

class InverseHeads(nn.Module):
    def __init__(self, spec: SpecBundle, params: ParamBundle, learnable_scale=True):
        super().__init__()
        self.emb_dim = spec.emb_dim
        self.meta = spec.meta
        # categorical prototypes
        self.register_buffer("W_opcode", torch.from_numpy(params.W_opcode).float() if params.W_opcode is not None else None)
        self.register_buffer("W_cid",    torch.from_numpy(params.W_cluster_id).float() if params.W_cluster_id is not None else None)
        self.scale_op  = nn.Parameter(torch.tensor(10.0)) if (learnable_scale and self.W_opcode is not None) else None
        self.scale_cid = nn.Parameter(torch.tensor(10.0)) if (learnable_scale and self.W_cid is not None) else None
        # scalar anchors
        self.scalar_cfg: Dict[str, Dict[str, torch.Tensor]] = {}
        for f in SCALAR_FEATS:
            d: Dict[str, torch.Tensor] = {}
            p = params.scalars.get(f, {})
            for k in ("e_min","e_max","table","bin_edges"):
                if k in p:
                    d[k] = torch.from_numpy(np.asarray(p[k])).float()
            self.scalar_cfg[f] = d
        # fallback projector
        self.proj = nn.Sequential(nn.Linear(self.emb_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    @staticmethod
    def _cosine_logits(Q: torch.Tensor, W: torch.Tensor, scale: Optional[torch.nn.Parameter]):
        Q = F.normalize(Q, dim=-1); Wn = F.normalize(W, dim=-1)
        L = Q @ Wn.T
        if scale is not None: L = L * scale
        return L

    @staticmethod
    def _project_alpha(e_hat: torch.Tensor, e_min: torch.Tensor, e_max: torch.Tensor):
        v = e_max - e_min
        num = (e_hat - e_min) @ v
        den = (v @ v).clamp_min(1e-8)
        return (num / den).clamp(0.0, 1.0)

    def _denorm_minmax(self, x01: torch.Tensor, feat: str):
        a = float(self.meta.get(f"min_{'local_offset' if feat=='offset' else feat}", 0.0))
        b = float(self.meta.get(f"max_{'local_offset' if feat=='offset' else feat}", 1.0))
        return a + x01 * (b - a)

    def _denorm_quantile(self, u: torch.Tensor, feat: str):
        cfg = self.scalar_cfg.get(feat, {})
        if ("table" in cfg) and ("bin_edges" in cfg):
            table = cfg["table"].to(u.device); edges = cfg["bin_edges"].to(u.device)
            j = torch.bucketize(u.clamp(0,1), edges) - 1
            j = j.clamp(0, edges.numel()-2)
            u0, u1 = edges[j], edges[j+1]
            x0, x1 = table[j], table[j+1]
            w = (u - u0) / (u1 - u0 + 1e-8)
            return x0 + w * (x1 - x0)
        return self._denorm_minmax(u, feat)

    def _postprocess(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in pred.items():
            if k == "length":
                unit = 4096.0
                out[k] = (torch.round(v / unit) * unit).clamp_min(0.0)
            elif k == "timediff":
                out[k] = v.clamp_min(0.0)
            else:
                out[k] = v
        return out

    def forward(self, emb_rec: torch.Tensor):
        B, C, H, D = emb_rec.shape
        e_cid = emb_rec[:,0]; e_cc=emb_rec[:,1]; e_off=emb_rec[:,2]; e_op=emb_rec[:,3]; e_len=emb_rec[:,4]; e_td=emb_rec[:,5]

        cat_logits: Dict[str, torch.Tensor] = {}
        if self.W_opcode is not None:
            Lop = self._cosine_logits(e_op.reshape(B*H, D), self.W_opcode, self.scale_op).view(B,H,-1)
            cat_logits["opcode"] = Lop
        else:
            cat_logits["opcode"] = None
        if self.W_cid is not None:
            Lcid= self._cosine_logits(e_cid.reshape(B*H, D), self.W_cid, self.scale_cid).view(B,H,-1)
            cat_logits["cluster_id"] = Lcid
        else:
            cat_logits["cluster_id"] = None

        def proj_u(E, feat: str):
            cfg = self.scalar_cfg.get(feat, {})
            if "e_min" in cfg and "e_max" in cfg:
                e_min = cfg["e_min"].to(emb_rec.device)
                e_max = cfg["e_max"].to(emb_rec.device)
                return self._project_alpha(E.reshape(B*H, D), e_min, e_max).view(B, H)
            return torch.sigmoid(self.proj(E.reshape(B*H, D))).view(B, H)

        u_cc  = proj_u(e_cc,  "cluster_center").clamp(0,1)
        u_off = proj_u(e_off, "offset").clamp(0,1)
        u_len = proj_u(e_len, "length").clamp(0,1)
        u_td  = proj_u(e_td,  "timediff").clamp(0,1)

        u_pred = {"cluster_center": u_cc, "offset": u_off, "length": u_len, "timediff": u_td}
        raw = {
            "offset":         self._denorm_minmax(u_off, "offset"),
            "cluster_center": self._denorm_quantile(u_cc,  "cluster_center"),
            "length":         self._denorm_quantile(u_len, "length"),
            "timediff":       self._denorm_quantile(u_td,  "timediff"),
        }
        raw = self._postprocess(raw)
        return cat_logits, u_pred, raw

class Emb2LatentAE(nn.Module):
    def __init__(self, spec: SpecBundle, params: ParamBundle, c_in=6, c_lat=8):
        super().__init__()
        self.backbone = AEBackbone(c_in, c_lat)
        self.heads    = InverseHeads(spec, params)
    def forward(self, x):
        z, skips = self.backbone.encode(x)
        emb_rec  = self.backbone.decode(z, skips)
        cat_logits, u_pred, raw = self.heads(emb_rec)
        return z, emb_rec, cat_logits, u_pred, raw

# ----------------------------
# 체크포인트 / 메트릭 / 플로팅
# ----------------------------
def find_latest_ckpt(ckpt_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "stage2_*.pt")))
    if not cands:
        # stage1 이름 규칙을 썼을 수 있어 백업 탐색
        cands = sorted(glob.glob(os.path.join(ckpt_dir, "stage2*.pt")))
    if not cands:
        raise FileNotFoundError(f"No stage2 checkpoint in {ckpt_dir}")
    return max(cands, key=os.path.getmtime)

@torch.no_grad()
def categorical_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    logits: (B,H,K), target: (B,H) with IGNORE_INDEX for invalid
    """
    if logits is None: 
        return float("nan")
    pred = logits.argmax(dim=-1)
    mask = (target != IGNORE_INDEX)
    correct = ((pred == target) & mask).sum().item()
    total   = mask.sum().item()
    return (correct / total) if total > 0 else float("nan")

@torch.no_grad()
def evaluate_batch(cat_logits: Dict[str, torch.Tensor],
                   y_cat: Dict[str, torch.Tensor],
                   raw_pred: Dict[str, torch.Tensor],
                   y_raw: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    # scalar
    for f in SCALAR_FEATS:
        yp = raw_pred[f]   # (B,H)
        yt = y_raw[f]      # (B,H)
        out[f"{f}.mae"] = F.l1_loss(yp, yt, reduction="mean").item()
        out[f"{f}.mse"] = F.mse_loss(yp, yt, reduction="mean").item()
    # categorical
    for f in CATEG_FEATS:
        if cat_logits.get(f) is not None:
            acc = categorical_accuracy(cat_logits[f], y_cat[f])
            out[f"{f}.acc"] = acc
        else:
            out[f"{f}.acc"] = float("nan")
    return out

def inverse_map_cluster_id(idx: torch.Tensor, cid_min: int) -> torch.Tensor:
    """
    학습시 1..K로 쉬프트된 것을 원래 cluster_id로 복구 (0 또는 IGNORE는 NaN으로 표기)
    """
    x = idx.clone().float()
    nan_mask = (x <= 0)  # 0 또는 음수(IGNORE_INDEX) → 복구 불가
    x = x - 1 + cid_min
    x[nan_mask] = float("nan")
    return x

def plot_one_window(tgt_6xH: Dict[str, np.ndarray],
                    pred_6xH: Dict[str, np.ndarray],
                    out_png: str):
    feats = FEATURES
    H = tgt_6xH[feats[0]].shape[0]
    t = np.arange(H)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=140)
    axes = axes.ravel()
    for i, f in enumerate(feats):
        ax = axes[i]
        y_true = tgt_6xH[f]
        y_pred = pred_6xH[f]
        if f in CATEG_FEATS:
            ax.step(t, y_true, where="mid", label="orig", linewidth=1.5)
            ax.step(t, y_pred, where="mid", label="gen",  linewidth=1.2)
        else:
            ax.plot(t, y_true, label="orig", linewidth=1.5)
            ax.plot(t, y_pred, label="gen",  linewidth=1.2)
        ax.set_title(f)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Stage-2 Raw Trace (blue=orig, orange=gen)")
    fig.tight_layout(rect=[0,0,1,0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"[plot] saved: {out_png}")

# ----------------------------
# 메인
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--csv-dir",   type=str, required=True)
    ap.add_argument("--spec",      type=str, required=True)
    ap.add_argument("--params-npz",type=str, required=True)
    ap.add_argument("--ckpt-dir",  type=str, required=True)
    ap.add_argument("--batch-size",type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--plot-out", type=str, default="./outputs_stage2/infer_win0.png")
    ap.add_argument("--metrics-out", type=str, default="./outputs_stage2/infer_metrics.npy")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec   = load_spec(args.spec)
    params = load_params_npz(args.params_npz)
    validate_spec_params(spec, params)

    ds = EmbeddingCSVWindowDataset(args.embed_dir, args.csv_dir, spec,
                                   win=args.win, stride=args.stride, dtype=torch.float32)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)

    model = Emb2LatentAE(spec, params).to(device)

    # stage2 체크포인트 로드
    ckpt_path = find_latest_ckpt(args.ckpt_dir)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"[ckpt] loaded: {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")

    # 평가 + 플롯(첫 윈도우 1개)
    model.eval()
    all_metrics: Dict[str, list] = {}
    first_done = False

    with torch.inference_mode():
        for x, y_cat, y_raw in dl:
            x = x.to(device, non_blocking=True)              # (B,6,512,256)
            y_cat = {k:v.to(device) for k,v in y_cat.items()}# (B,512)
            y_raw = {k:v.to(device) for k,v in y_raw.items()}# (B,512)

            z, emb_rec, cat_logits, u_pred, raw_pred = model(x)

            # 메트릭 집계
            m = evaluate_batch(cat_logits, y_cat, raw_pred, y_raw)
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)

            # 첫 번째 배치의 첫 윈도우를 그림 저장
            if not first_done:
                B = x.shape[0]
                i = 0  # 첫 샘플
                H = x.shape[2]

                # 원본: 범주형(인덱스) / 스칼라(실수)
                tgt = {
                    "cluster_id": inverse_map_cluster_id(y_cat["cluster_id"][i], spec.cluster_id_min).cpu().numpy(),
                    "cluster_center": y_raw["cluster_center"][i].cpu().numpy(),
                    "offset":         y_raw["offset"][i].cpu().numpy(),
                    "opcode":         y_cat["opcode"][i].float().cpu().numpy(),
                    "length":         y_raw["length"][i].cpu().numpy(),
                    "timediff":       y_raw["timediff"][i].cpu().numpy(),
                }

                # 생성: 범주형 argmax → (원래 id로 역변환), 스칼라 raw 예측
                gen_cid_idx = cat_logits["cluster_id"][i].argmax(dim=-1) if cat_logits["cluster_id"] is not None else torch.full((H,), float("nan"))
                gen_op_idx  = cat_logits["opcode"][i].argmax(dim=-1)     if cat_logits["opcode"] is not None else torch.full((H,), float("nan"))
                pred = {
                    "cluster_id": inverse_map_cluster_id(gen_cid_idx, spec.cluster_id_min).cpu().numpy() if cat_logits["cluster_id"] is not None else np.full((H,), np.nan, dtype=float),
                    "cluster_center": raw_pred["cluster_center"][i].cpu().numpy(),
                    "offset":         raw_pred["offset"][i].cpu().numpy(),
                    "opcode":         gen_op_idx.float().cpu().numpy() if cat_logits["opcode"] is not None else np.full((H,), np.nan, dtype=float),
                    "length":         raw_pred["length"][i].cpu().numpy(),
                    "timediff":       raw_pred["timediff"][i].cpu().numpy(),
                }

                plot_one_window(tgt, pred, args.plot_out)
                first_done = True

    # 평균 메트릭 요약/저장
    summary = {k: float(np.nanmean(v)) for k, v in all_metrics.items()}
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    np.save(args.metrics_out, summary)
    print("[metrics] " + " | ".join([f"{k}={v:.6f}" for k,v in summary.items()]))
    print(f"[metrics] saved: {args.metrics_out}")

if __name__ == "__main__":
    main()
