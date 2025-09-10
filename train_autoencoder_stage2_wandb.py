#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2: Joint Learning (Embedding + Raw CSV Reconstruction)

Inputs:
  - Embedding windows: (B,6,512,256)
  - Raw labels from CSV: opcode_id, cluster_id, {cluster_center, offset, length, timediff}

Loss:
  L = λ_recon * MSE(emb_rec, emb_gt)
    + λ_cat   * [CE(opcode) + CE(cluster_id)]
    + λ_sraw  * sum_f MAE(raw_pred[f], raw_label[f])
    + λ_u     * (optional; disabled by default)



python train_autoencoder_stage2.py \
  --embed-dir out/emb_1 \
  --csv-dir   out/emb_1/cluster\
  --spec      out/emb_1/rowemb_spec.json \
  --params-npz out/emb_1/rowemb_params.npz\
  --init-from ./outputs_stage1/stage1_0210000.pt \
  --w-recon 0.5 --w-cat 0.7 --w-sraw 0.7 \
  --batch-size 4 --lr 1e-4
"""
from __future__ import annotations

import os, json, csv, glob, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch.cuda.amp import autocast, GradScaler


import wandb  # ← wandb 통합

# ----------------------------
# Constants / Features
# ----------------------------

FEATURES = ["cluster_id", "cluster_center", "offset", "opcode", "length", "timediff"]
SCALAR_FEATS = ["cluster_center", "offset", "length", "timediff"]
CATEG_FEATS  = ["opcode", "cluster_id"]

IGNORE_INDEX = -100  # CE에서 무시할 타깃 인덱스

# ----------------------------
# Spec / Params
# ----------------------------

@dataclass
class SpecBundle:
    opcode_vocab: List[str]
    cluster_id_min: int
    cluster_id_max: int
    emb_dim: int
    meta: Dict[str, Any]

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

@dataclass
class ParamBundle:
    W_opcode: np.ndarray | None
    W_cluster_id: np.ndarray | None
    scalars: Dict[str, Dict[str, np.ndarray]]

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
        assert d == spec.emb_dim, "opcode weight dim mismatch"
        assert r == len(spec.opcode_vocab), "opcode rows vs vocab mismatch"
    if params.W_cluster_id is not None:
        r, d = params.W_cluster_id.shape
        assert d == spec.emb_dim, "cluster_id weight dim mismatch"
        expected = (spec.cluster_id_max - spec.cluster_id_min + 1) + 1
        assert r == expected, f"cluster_id rows {r} != expected {expected}"
    for feat in SCALAR_FEATS:
        sc = params.scalars.get(feat, {})
        if "e_min" in sc and "e_max" in sc:
            assert sc["e_min"].shape == (spec.emb_dim,)
            assert sc["e_max"].shape == (spec.emb_dim,)

# ----------------------------
# Dataset with labels
# ----------------------------

def list_npy_files(embed_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy found in {embed_dir}")
    return files

def default_csv_for_npy(npy_path: str, csv_dir: str) -> str:
    base = os.path.splitext(os.path.basename(npy_path))[0]
    cand = os.path.join(csv_dir, base + ".csv")
    if not os.path.isfile(cand):
        raise FileNotFoundError(f"CSV not found for {npy_path}: expect {cand}")
    return cand

def build_op2id(vocab: List[str]) -> Dict[str, int]:
    return {op: i for i, op in enumerate(vocab)}

def map_opcode(op: str, op2id: Dict[str,int]) -> int:
    # vocab에 없으면 ignore
    return op2id.get(op, IGNORE_INDEX)

def map_cluster_id(cid_val: int, cid_min: int, cid_max: int) -> int:
    # [min,max] → [1..K], 범위 밖은 ignore_index
    if cid_min <= cid_val <= cid_max:
        return (cid_val - cid_min) + 1
    return IGNORE_INDEX


class WindowSample:
    def __init__(self, emb, opcode_id, cluster_id, scalars):
        self.emb = emb
        self.opcode_id = opcode_id
        self.cluster_id = cluster_id
        self.scalars = scalars


# --- CSV 라벨 캐시 ---
class _CSVLabelCache:
    def __init__(self, cache_dir: str | None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.mem: Dict[str, Dict[str, np.ndarray]] = {}

    def _cache_path(self, csv_path: str) -> str:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        return os.path.join(self.cache_dir, base + ".npz") if self.cache_dir else ""

    def load_or_build(
        self, csv_path: str, op2id: Dict[str,int], cid_min: int, cid_max: int
    ) -> Dict[str, np.ndarray]:
        if csv_path in self.mem:
            return self.mem[csv_path]

        # 1) 디스크 캐시
        if self.cache_dir:
            cp = self._cache_path(csv_path)
            if os.path.isfile(cp):
                arrs = np.load(cp)
                obj = {k: arrs[k] for k in arrs.files}
                self.mem[csv_path] = obj
                return obj

        # 2) CSV 파싱 → NumPy
        op_ids, cids = [], []
        cc, off, leng, td = [], [], [], []
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
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

        # 3) 디스크 캐시 저장
        if self.cache_dir:
            np.savez_compressed(self._cache_path(csv_path), **obj)

        self.mem[csv_path] = obj
        return obj
    

class EmbeddingCSVWindowDataset(Dataset):
    """
    embed_dir 내 *.npy: (N,6,256)을 읽어 슬라이딩 윈도우(6,512,256)로 꺼냄.
    csv_dir 내 *.csv: 동일 구간 [s:e) 라벨을 y_cat(H), y_raw(H)로 반환함.
    """
    def __init__(self, embed_dir, csv_dir, spec, win=512, stride=256, dtype=torch.float32,
                 label_cache_dir: str | None = "out/label_cache"):
        super().__init__()
        self.embed_dir = embed_dir
        self.csv_dir   = csv_dir
        self.spec      = spec
        self.win       = int(win)
        self.stride    = int(stride)
        self.dtype     = dtype

        # ---- vocab/범위 ----
        self.op2id   = build_op2id(list(spec.opcode_vocab))
        self.cid_min = int(spec.cluster_id_min)
        self.cid_max = int(spec.cluster_id_max)

        # ---- 파일 인덱싱 ----
        self.embed_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))
        if not self.embed_files:
            raise FileNotFoundError(f"No .npy in {embed_dir}")

        self.csv_files   = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        if not self.csv_files:
            raise FileNotFoundError(f"No .csv in {csv_dir}")

        self.file_meta: List[Dict] = []  # [{npy, csv, N}, ...]
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

        # ---- CSV 라벨 캐시 미리 적재 ----
        self._label_cache = _CSVLabelCache(label_cache_dir)
        for meta in self.file_meta:
            _ = self._label_cache.load_or_build(
                meta["csv"], self.op2id, self.cid_min, self.cid_max
            )

        # ---- 임베딩 memmap 핸들 (lazy) ----
        self._emb_mm: Dict[str, np.memmap] = {}

        # ---- 윈도우 인덱스 생성 ----
        self.index: List[Tuple[int,int]] = []
        for fi, meta in enumerate(self.file_meta):
            N = meta["N"]
            if N < self.win:
                continue
            for s in range(0, N - self.win + 1, self.stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"No windows with win={self.win}, stride={self.stride}")

    def __len__(self):
        return len(self.index)

    def _emb_arr(self, npy_path: str) -> np.memmap:
        arr = self._emb_mm.get(npy_path)
        if arr is None:
            arr = np.load(npy_path, mmap_mode="r")  # (N,6,256)
            self._emb_mm[npy_path] = arr
        return arr

    def __getitem__(self, idx: int):
        fi, s = self.index[idx]
        e = s + self.win
        meta = self.file_meta[fi]

        # ---- 임베딩 윈도우 ----
        arr = self._emb_arr(meta["npy"])           # (N,6,256)
        win_np = arr[s:e]                           # (H,6,256)
        emb = torch.from_numpy(win_np.transpose(1,0,2).copy()).to(self.dtype)

        # ---- 라벨 슬라이스 ----
        lbl_np = self._label_cache.mem[meta["csv"]]
        y_cat = {
            "opcode":     torch.from_numpy(lbl_np["opcode"][s:e]).long(),      # (H,)
            "cluster_id": torch.from_numpy(lbl_np["cluster_id"][s:e]).long(),  # (H,)
        }
        y_raw = {
            "cluster_center": torch.from_numpy(lbl_np["cluster_center"][s:e]).float(),
            "offset":         torch.from_numpy(lbl_np["offset"][s:e]).float(),
            "length":         torch.from_numpy(lbl_np["length"][s:e]).float(),
            "timediff":       torch.from_numpy(lbl_np["timediff"][s:e]).float(),
        }

        return emb, y_cat, y_raw


# ----------------------------
# Backbone (same as Stage-1)
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        groups = 8
        if c_out % groups != 0:
            for g in (4, 2, 1):
                if c_out % g == 0:
                    groups = g; break
        self.norm = nn.GroupNorm(groups, c_out)
        self.act  = nn.SiLU()
    def forward(self, x): 
        return self.act(self.norm(self.conv(x)))


class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.b1 = ConvBlock(c_in, c_out)
        self.b2 = ConvBlock(c_out, c_out)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        h = self.b2(self.b1(x))   # same H,W
        y = self.down(h)          # H/2, W/2
        return y, h               # y(next), h(skip at input H,W)

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.b1 = ConvBlock(c_out + c_skip, c_out)
        self.b2 = ConvBlock(c_out, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(f"Up mismatch x={x.shape}, skip={skip.shape}")
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
        # Enc: 512x256 -> 256x128 -> 128x64 -> 64x32
        self.down1 = Down(c_in, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, c_lat)
        self.mid   = ConvBlock(c_lat, c_lat)
        # Dec: 64x32 -> 128x64 -> 256x128 -> 512x256
        self.up3   = Up(c_in=c_lat, c_skip=128, c_out=128)  # skip: x2
        self.up2   = Up(c_in=128,   c_skip=64,  c_out=64)   # skip: x1
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

# ----------------------------
# Inverse Heads (categorical + scalar)
# ----------------------------

class InverseHeads(nn.Module):
    def __init__(self, spec: SpecBundle, params: ParamBundle, learnable_scale=True):
        super().__init__()
        self.emb_dim = spec.emb_dim
        self.meta = spec.meta

        # categorical prototypes
        self.register_buffer("W_opcode", torch.from_numpy(params.W_opcode).float() if params.W_opcode is not None else None)
        self.register_buffer("W_cid",    torch.from_numpy(params.W_cluster_id).float() if params.W_cluster_id is not None else None)
        self.scale_op = nn.Parameter(torch.tensor(10.0)) if (learnable_scale and self.W_opcode is not None) else None
        self.scale_cid= nn.Parameter(torch.tensor(10.0)) if (learnable_scale and self.W_cid is not None) else None

        # scalar anchors/tables
        self.scalar_cfg: Dict[str, Dict[str, torch.Tensor]] = {}
        for f in SCALAR_FEATS:
            d: Dict[str, torch.Tensor] = {}
            p = params.scalars.get(f, {})
            for k in ("e_min","e_max","table","bin_edges"):
                if k in p:
                    d[k] = torch.from_numpy(np.asarray(p[k])).float()
            self.scalar_cfg[f] = d

        # learnable projector fallback (if no anchors)
        self.proj = nn.Sequential(nn.Linear(self.emb_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    # u ← raw (min–max 역정규화의 역함수; 학습용)
    @staticmethod
    def _norm_minmax_inv(x_raw: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
        u = (x_raw - x_min) / max(1e-12, (x_max - x_min))
        return u.clamp(0, 1)

    # u ← raw (quantile 역매핑; 학습용)
    def _quantile_inv(self, x_raw: torch.Tensor, feat: str) -> torch.Tensor:
        cfg = self.scalar_cfg[feat]
        table = cfg["table"].to(x_raw.device)        # (K,)
        edges = cfg["bin_edges"].to(x_raw.device)    # (K,) or (K+1,)
        K = table.shape[0]

        xr = x_raw.reshape(-1)
        idx = torch.searchsorted(table, xr, right=False).clamp(1, K-1)
        i0 = idx - 1
        i1 = idx

        t0 = table[i0]
        t1 = table[i1]
        denom = (t1 - t0).clamp_min(1e-12)
        alpha = (xr - t0) / denom
        alpha = alpha.clamp(0, 1)

        if edges.numel() == K + 1:
            u0 = edges[i0]; u1 = edges[i1]
        else:
            step = 1.0 / float(K - 1)
            u0 = i0.float() * step
            u1 = i1.float() * step

        u = (1 - alpha) * u0 + alpha * u1
        return u.view_as(x_raw).clamp(0, 1)

    @staticmethod
    def _cosine_logits(Q: torch.Tensor, W: torch.Tensor, scale: torch.nn.Parameter | None):
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

    # raw ← u (추론 경로)
    def _denorm_minmax(self, x01: torch.Tensor, feat: str):
        a = float(self.meta.get(f"min_{'local_offset' if feat=='offset' else feat}", 0.0))
        b = float(self.meta.get(f"max_{'local_offset' if feat=='offset' else feat}", 1.0))
        return a + x01 * (b - a)

    def _denorm_quantile(self, u: torch.Tensor, feat: str):
        cfg = self.scalar_cfg.get(feat, {})
        if "table" in cfg and "bin_edges" in cfg:
            table = cfg["table"]; edges = cfg["bin_edges"]
            u = u.clamp(0.0, 1.0)
            j = torch.bucketize(u, edges) - 1
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

        # categorical
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

        # scalars: u projection
        def proj_u(E, feat: str):
            cfg = self.scalar_cfg.get(feat, {})
            if "e_min" in cfg and "e_max" in cfg:
                e_min = cfg["e_min"].to(emb_rec.device, non_blocking=True)
                e_max = cfg["e_max"].to(emb_rec.device, non_blocking=True)
                return self._project_alpha(E.reshape(B*H, D), e_min, e_max).view(B, H)
            return torch.sigmoid(self.proj(E.reshape(B*H, D))).view(B, H)

        u_cc  = proj_u(e_cc,  "cluster_center")
        u_off = proj_u(e_off, "offset")
        u_len = proj_u(e_len, "length")
        u_td  = proj_u(e_td,  "timediff")

        u_pred: Dict[str, torch.Tensor] = {
            "cluster_center": u_cc.clamp(0, 1),
            "offset":         u_off.clamp(0, 1),
            "length":         u_len.clamp(0, 1),
            "timediff":       u_td.clamp(0, 1),
        }

        # denorm to raw (추론)
        raw: Dict[str, torch.Tensor] = {}
        raw["offset"]         = self._denorm_minmax(u_pred["offset"],           "offset")
        raw["cluster_center"] = self._denorm_quantile(u_pred["cluster_center"], "cluster_center")
        raw["length"]         = self._denorm_quantile(u_pred["length"],         "length")
        raw["timediff"]       = self._denorm_quantile(u_pred["timediff"],       "timediff")
        raw = self._postprocess(raw)

        return cat_logits, u_pred, raw


# ----------------------------
# Full model
# ----------------------------

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
# Loss
# ----------------------------

@dataclass
class Weights:
    w_recon: float = 1.0
    w_cat: float = 1.0
    w_sraw: float = 1.0
    w_kl: float = 0.0 # KL지금 안씀


def compute_losses(emb_rec, emb_gt, cat_logits, y_cat, u_pred, raw_pred, y_raw,
                   weights, heads, feat_weights: dict[str, float] | None = None):
    """
    feat_weights: 각 feature별 동적 가중치(dict). 없으면 1.0로 처리함.
    """
    IGN = -100
    L_recon = F.mse_loss(emb_rec, emb_gt)

    # --- categorical ---
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGN)
    per_cat_loss: Dict[str, float | None] = {}
    L_cat = torch.tensor(0.0, device=emb_rec.device)
    if cat_logits.get("opcode") is not None:
        loss_opcode = ce(cat_logits["opcode"].transpose(1, 2), y_cat["opcode"].long())
        L_cat = L_cat + loss_opcode
        per_cat_loss["L_opcode(CE)"] = float(loss_opcode.item())
    else:
        per_cat_loss["L_opcode(CE)"] = None
    if cat_logits.get("cluster_id") is not None:
        loss_cluster = ce(cat_logits["cluster_id"].transpose(1, 2), y_cat["cluster_id"].long())
        L_cat = L_cat + loss_cluster
        per_cat_loss["L_cluster_id(CE)"] = float(loss_cluster.item())
    else:
        per_cat_loss["L_cluster_id(CE)"] = None

    # --- scalar: u-gt 생성 함수 (기존 경로 그대로) ---
    def get_minmax_from_meta(feat: str):
        name = "local_offset" if feat == "offset" else feat
        a = float(heads.meta.get(f"min_{name}", np.nan))
        b = float(heads.meta.get(f"max_{name}", np.nan))
        if not np.isfinite(a) or not np.isfinite(b):
            return None, None
        lo = torch.tensor(a, device=emb_rec.device, dtype=torch.float32)
        hi = torch.tensor(b, device=emb_rec.device, dtype=torch.float32)
        return lo, hi

    def raw_to_u_gt(feat: str, x_raw: torch.Tensor) -> Optional[torch.Tensor]:
        cfg = heads.scalar_cfg.get(feat, {})
        if ("table" in cfg) and ("bin_edges" in cfg):
            return heads._quantile_inv(x_raw, feat)
        lo, hi = get_minmax_from_meta(feat)
        if (lo is not None) and (hi is not None):
            return InverseHeads._norm_minmax_inv(x_raw, lo, hi)
        return None

    # --- feature별 손실 설계 & 가중 합 ---
    per_feat_loss_u: Dict[str, float | None] = {}
    per_feat_loss_raw: Dict[str, float | None] = {}
    L_sraw_weighted = torch.tensor(0.0, device=emb_rec.device)
    used = 0

    # 기본 가중치 디폴트 1.0
    if feat_weights is None:
        feat_weights = {f: 1.0 for f in ("cluster_center","offset","length","timediff")}

    for feat in ("cluster_center", "offset", "length", "timediff"):
        ugt = raw_to_u_gt(feat, y_raw[feat])
        if (ugt is not None) and (feat in u_pred) and (u_pred[feat] is not None):
            # 공통: u-space L1
            L_u = F.l1_loss(u_pred[feat], ugt)
            per_feat_loss_u[f"L_{feat}(u)"] = float(L_u.item())

            # 보조: raw-space 보조 손실(선택적)
            L_raw_aux = None
            if feat == "timediff":
                # heavy-tail 안정화: log1p(raw) L1
                td_pred = raw_pred["timediff"].clamp_min(0)
                td_gt   = y_raw["timediff"].clamp_min(0)
                L_raw_aux = F.l1_loss(torch.log1p(td_pred), torch.log1p(td_gt))
                per_feat_loss_raw["L_timediff_raw_log1p"] = float(L_raw_aux.item())

                # 가중 합: u L1 + α * log1p raw L1
                alpha = 0.5  # 시작값; 필요시 조정
                L_feat = L_u + alpha * L_raw_aux
            else:
                # length/offset/cluster_center는 u L1만 기본 사용(원하면 raw 보조 추가 가능)
                L_feat = L_u

            w_f = float(feat_weights.get(feat, 1.0))
            L_sraw_weighted = L_sraw_weighted + w_f * L_feat
            used += 1
        else:
            per_feat_loss_u[f"L_{feat}(u)"] = None
            if feat == "timediff":
                per_feat_loss_raw["L_timediff_raw_log1p"] = None

    if used == 0:
        L_sraw_weighted = torch.tensor(0.0, device=emb_rec.device)

    # --- total ---
    L = (weights.w_recon * L_recon
         + weights.w_cat   * L_cat
         + weights.w_sraw  * L_sraw_weighted)

    logs = {
        "L_total": float(L.item()),
        "L_recon": float(L_recon.item()),
        "L_cat_total": float(L_cat.item()),
        "L_sraw_weighted": float(L_sraw_weighted.item()),
        "scalar_terms_used": used,
    }
    logs.update(per_cat_loss)
    logs.update(per_feat_loss_u)
    logs.update(per_feat_loss_raw)
    # 현재 step의 feature별 가중치도 로깅
    for f, w in feat_weights.items():
        logs[f"w_{f}"] = float(w)
    return L, logs

# ----------------------------
# Artifact logging & rotation
# ----------------------------

def _log_ckpt_and_rotate_artifacts(ckpt_path: str, artifact_name: str = "stage2-model", epoch: int | None = None):
    """
    ckpt_path를 artifact로 업로드하고, 동일 이름의 model artifacts 중 최신 2개만 남기고 나머지 삭제함.
    - WANDB_API_KEY가 설정돼 있어야 삭제 가능함.
    - 실패하더라도 학습은 계속 진행됨.
    """
    try:
        aliases = [f"epoch-{epoch}"] if epoch is not None else []
        # 새 아티팩트 업로드(새 버전 생성). 'latest' alias를 부여하면 보통 가장 최근 버전으로 이동함.
        artifact = wandb.Artifact(artifact_name, type="model", metadata={"epoch": epoch})
        artifact.add_file(ckpt_path, name=os.path.basename(ckpt_path))
        logged = wandb.log_artifact(artifact, aliases=aliases + ["latest"])
        # 업로드가 끝나길 대기(선택)
        try:
            logged.wait()
        except Exception:
            pass

        # 회전: 최신 2개만 유지
        try:
            api = wandb.Api()
            entity, project = wandb.run.entity, wandb.run.project
            versions = api.artifact_versions("model", f"{entity}/{project}/{artifact_name}")
            # versions는 최신순으로 정렬되어 옴(보통)
            for i, art in enumerate(versions):
                if i == 0:
                    # 최신: latest alias 유지, 필요시 prev 제거
                    pass
                elif i == 1:
                    # 두 번째 최신: prev alias가 없으면 붙여줌(실패해도 무시)
                    try:
                        cur_aliases = set(art.aliases or [])
                        if "prev" not in cur_aliases:
                            art.aliases.append("prev")
                            art.save()
                    except Exception:
                        pass
                else:
                    # 나머지 오래된 것 삭제
                    try:
                        art.delete()
                    except Exception:
                        pass
        except Exception:
            # API 접근 실패 시 조용히 넘어감
            pass

    except Exception as e:
        print(f"[wandb][artifact] failed to log/rotate: {e}")

# ----------------------------
# Train
# ----------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec   = load_spec(args.spec)
    params = load_params_npz(args.params_npz)
    validate_spec_params(spec, params)

    ds = EmbeddingCSVWindowDataset(args.embed_dir, args.csv_dir, spec,
                                   win=args.win, stride=args.stride, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

    model = Emb2LatentAE(spec, params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=args.amp)   # <-- AMP 스케일러

    weights = Weights(w_recon=args.w_recon, w_cat=args.w_cat, w_sraw=args.w_sraw)

    # Stage-1 가중치 로드
    if args.init_from and os.path.isfile(args.init_from):
        try:
            ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"[init] loaded with missing={len(missing)}, unexpected={len(unexpected)}")

    # --- wandb Init ---
    wandb.init(
        project="ditto-stage2",
        name=f"run_{os.path.basename(args.out_dir)}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "w_recon": args.w_recon,
            "w_cat": args.w_cat,
            "w_sraw": args.w_sraw,
            "win": args.win,
            "stride": args.stride,
        }
    )
    wandb.watch(model, log="all", log_freq=100)

    ema = {f: None for f in ("cluster_center","offset","length","timediff")}
    beta = 0.9           # EMA 계수
    gamma = 1.0          # 가중치 민감도(>1이면 더 공격적으로 높임)
    min_floor = 1e-6     # 수치 안정


    num_epochs = args.epochs if hasattr(args, "epochs") else 256
    global_step = 0  # ← 누락되어 있던 전역 스텝 초기화

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        feat_w = {f: 1.0 for f in ("cluster_center","offset","length","timediff")}  # 초기 가중치


        for step, (emb, y_cat, y_raw) in enumerate(dl, start=1):
            emb = emb.to(device)
            y_cat = {k:v.to(device) for k,v in y_cat.items()}
            y_raw = {k:v.to(device) for k,v in y_raw.items()}

            with autocast(enabled=args.amp, dtype=torch.bfloat16):
                z, emb_rec, cat_logits, u_pred, raw_pred = model(emb)
                L, logs = compute_losses(emb_rec, emb, cat_logits, y_cat, u_pred, raw_pred,
                                    y_raw, weights, model.heads, feat_weights=feat_w)

            opt.zero_grad(set_to_none=True)
            # AMP backward
            scaler.scale(L).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()


            # === EMA 업데이트(각 feature의 u-space 손실 기준; timediff는 보조 항이 logs에 따로 있음) ===
            with torch.no_grad():
                for f in ("cluster_center","offset","length","timediff"):
                    key = f"L_{f}(u)"
                    val = logs.get(key, None)
                    if val is None or not np.isfinite(val): 
                        continue
                    ema[f] = (beta*ema[f] + (1-beta)*val) if (ema[f] is not None) else val

                # === 동적 가중치 갱신 ===
                # 전략: ema가 큰 feature일수록 가중치를 크게.
                #   w_f_raw = (ema[f] / (min_ema + eps)) ** gamma
                em_vals = [v for v in ema.values() if v is not None]
                if em_vals:
                    min_ema = max(min(em_vals), min_floor)
                    w_raw = {f: ((ema[f] / min_ema) ** gamma if ema[f] is not None else 1.0)
                            for f in ("cluster_center","offset","length","timediff")}
                    # 정규화: 평균이 1이 되도록 → 전체 스케일 유지 (w_sraw만으로 총량 제어)
                    mean_w = np.mean([w_raw[f] for f in w_raw])
                    feat_w = {f: (w_raw[f] / mean_w) for f in w_raw}

                    # wandb에 현재 step의 동적 가중치 로그(추적용)
                    wandb.log({f"w_{f}": feat_w[f] for f in feat_w}, step=global_step)

            epoch_loss += L.item()
            global_step += 1

            # step logging
            wandb.log(logs, step=global_step)
            wandb.log({"lr": opt.param_groups[0]["lr"]}, step=global_step)

        # epoch logging
        avg_loss = epoch_loss / len(dl)
        wandb.log({"avg_loss": avg_loss, "epoch": epoch}, step=global_step)

        if epoch % args.log_every == 0:
            print(f"[epoch {epoch} | step {step}] " +
                  " | ".join([f"{k}:{(v if v is not None else float('nan')):.4f}" for k,v in logs.items() if isinstance(v, (int,float))]))

        print(f"==== [Epoch {epoch}/{num_epochs}] Avg Loss: {avg_loss:.4f} ====")

        # --- epoch 체크포인트 저장 + wandb artifact (회전 유지: 최신 2개만) ---
        if (epoch % args.ckpt_every) == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"stage2_e{epoch:03d}.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)
            print("[ckpt][epoch] saved:", ckpt_path)

            # wandb artifact 업로드 + 회전
            _log_ckpt_and_rotate_artifacts(ckpt_path, artifact_name="stage2-model", epoch=epoch)

    print("Stage-2 training done.")
    os.makedirs(args.out_dir, exist_ok=True)
    final_ckpt = os.path.join(args.out_dir, "stage2_final.pt")
    torch.save({"model": model.state_dict()}, final_ckpt)
    print("[ckpt][final] saved:", final_ckpt)

    # 최종 모델도 artifact로 업로드 및 회전 정책 동일 적용
    _log_ckpt_and_rotate_artifacts(final_ckpt, artifact_name="stage2-model", epoch=num_epochs)


# ----------------------------
# Collate (moved here to avoid forward refs)
# ----------------------------

def collate_fn(batch):
    xs, ycats, yraws = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,6,512,256)

    def stack_long(key):
        return torch.stack([yc[key] for yc in ycats], dim=0).long()    # (B,H)

    def stack_float(key):
        return torch.stack([yr[key] for yr in yraws], dim=0).float()   # (B,H)

    y_cat = {
        "opcode":     stack_long("opcode"),
        "cluster_id": stack_long("cluster_id"),
    }
    y_raw = {
        "cluster_center": stack_float("cluster_center"),
        "offset":         stack_float("offset"),
        "length":         stack_float("length"),
        "timediff":       stack_float("timediff"),
    }
    return x, y_cat, y_raw


# ----------------------------
# Args
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--csv-dir", type=str, required=True)
    ap.add_argument("--spec", type=str, required=True)
    ap.add_argument("--params-npz", type=str, required=True)
    ap.add_argument("--init-from", type=str, default=None, help="optional stage-1 ckpt")
    ap.add_argument("--epochs", type=int, default=256, help="number of training epochs")

    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--w-recon", type=float, default=0.5)
    ap.add_argument("--w-cat", type=float, default=0.7)
    ap.add_argument("--w-sraw", type=float, default=0.7)

    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=16,
                    help="save checkpoint every N epochs (default: 16)")
    ap.add_argument("--out-dir", type=str, default="./outputs_stage2")
    ap.add_argument("--amp", action="store_true", help="Enable AMP (mixed precision) training with bfloat16 on GPU")

    return ap.parse_args()

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
