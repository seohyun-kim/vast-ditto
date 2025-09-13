#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1: Embedding Reconstruction Only

Input  : (B, 6, 512, 128) windows from .npy (N,6,128) with sliding windows
Target : same embedding tensor (autoencoder reconstruction)
Loss   : L = λ_recon * MSE(emb_rec, emb_gt)



python train_autoencoder_stage1.py   --embed-dir out/emb_1   --spec out/emb_1/rowemb_spec.json   --batch-size 4  --lr 1e-4

"""

from __future__ import annotations
import os, json, glob, csv, argparse, re

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ----------------------------
# Common utils
# ----------------------------

FEATURE_ORDER = ["cluster_id", "cluster_center", "offset", "opcode", "length", "timediff"]

def list_npy_files(embed_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy found in {embed_dir}")
    return files

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
        emb_dim=int(s.get("emb_dim", 128)),
        meta=dict(s.get("meta", {})),
    )

def validate_spec(spec: SpecBundle):
    assert spec.emb_dim == 128, f"emb_dim must be 128, got {spec.emb_dim}"

# ----------------------------
# Dataset (Stage-1: embedding only)
# ----------------------------

class EmbWindowDataset(torch.utils.data.Dataset):
    """
    Loads (N,6,128) memmap and yields windows as (6,512,128)
    """
    def __init__(self, embed_dir: str, win: int = 512, stride: int = 256, dtype=torch.float32):
        super().__init__()
        self.embed_files = list_npy_files(embed_dir)
        self.win = win
        self.stride = stride
        self.dtype = dtype

        self.index: List[Tuple[int,int]] = []  # (file_idx, start_row)
        self.file_meta: List[Dict[str, Any]] = []
        for fi, npy in enumerate(self.embed_files):
            arr = np.load(npy, mmap_mode="r")  # (N,6,128)
            if arr.ndim != 3 or arr.shape[1] != 6 or arr.shape[2] != 128:
                raise ValueError(f"Bad shape in {npy}: {arr.shape} expected (N,6,128)")
            N = arr.shape[0]
            self.file_meta.append({"npy": npy, "N": N})
            for st in range(0, max(0, N - win + 1), stride):
                self.index.append((fi, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, st = self.index[idx]
        meta = self.file_meta[fi]
        arr = np.load(meta["npy"], mmap_mode="r")
        win = arr[st:st+self.win]                # (512,6,128)
        # emb = torch.from_numpy(win.transpose(1,0,2)).to(self.dtype)  # (6,512,128)
        emb = torch.from_numpy(win.transpose(1,0,2).copy()).to(self.dtype)
        # .copy()를 붙여 writable 배열로 만든 뒤 텐서 변환
        return emb, emb.clone()  # input, target (reconstruction)

# ----------------------------
# SD-style lightweight AE backbone
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        # Choose the largest valid num_groups among [8, 4, 2, 1]
        groups = 8
        if c_out % groups != 0:
            for g in (4, 2, 1):
                if c_out % g == 0:
                    groups = g
                    break
        self.norm = nn.GroupNorm(groups, c_out)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# 1) 블록 일반화
class Down(nn.Module):
    def __init__(self, c_in, c_out, s=(2,2), k=(4,4), p=(1,1)):
        super().__init__()
        self.b1 = ConvBlock(c_in, c_out)
        self.b2 = ConvBlock(c_out, c_out)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=k, stride=s, padding=p)  # ← 기존 (4,4),(2,2),(1,1) 기본값과 동일:contentReference[oaicite:2]{index=2}
    def forward(self, x):
        h = self.b2(self.b1(x))
        y = self.down(h)
        return y, h

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out, s=(2,2), k=(4,4), p=(1,1), op=(0,0)):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, output_padding=op)  # 기본은 기존과 동일:contentReference[oaicite:3]{index=3}
        self.b1 = ConvBlock(c_out + c_skip, c_out)
        self.b2 = ConvBlock(c_out, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.b1(x)
        x = self.b2(x)
        return x

class UpNoSkip(nn.Module):
    def __init__(self, c_in, c_out, s=(2,2), k=(4,4), p=(1,1), op=(0,0)):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, output_padding=op)
        self.b1 = ConvBlock(c_out, c_out)
        self.b2 = ConvBlock(c_out, c_out)
    def forward(self, x):
        x = self.up(x)
        x = self.b1(x)
        x = self.b2(x)
        return x

# 2) AEBackbone에서 down3/up3만 비등방 설정
class AEBackbone(nn.Module):
    def __init__(self, c_in=6, c_lat=8):
        super().__init__()
        # Encoder
        self.down1 = Down(c_in, 64,  s=(2,2), k=(4,4), p=(1,1))
        self.down2 = Down(64, 128,  s=(2,2), k=(4,4), p=(1,1))
        self.down3 = Down(128, c_lat, s=(2,1), k=(4,3), p=(1,1))  # ← W 보존 (32)

        self.mid   = ConvBlock(c_lat, c_lat)

        # Decoder
        self.up3   = Up(c_in=c_lat, c_skip=128, c_out=128, s=(2,1), k=(4,3), p=(1,1), op=(0,0))  # ← x2와 해상도 정합
        self.up2   = Up(c_in=128,   c_skip=64,  c_out=64,  s=(2,2), k=(4,4), p=(1,1))
        self.up1   = UpNoSkip(c_in=64, c_out=64, s=(2,2), k=(4,4), p=(1,1))
        self.final = nn.Conv2d(64, c_in, kernel_size=1)

    def forward(self, x):  # x: (B,6,512,128)
        x1, _ = self.down1(x)     # (B,64,256,64)
        x2, _ = self.down2(x1)    # (B,128,128,32)
        x3, _ = self.down3(x2)    # (B,8,64,32)  ← latent 입력
        z = self.mid(x3)          # (B,8,64,32)  ← latent
        # print(z.shape, "latent vector shape")

        u2 = self.up3(z, x2)      # (B,128,128,32)
        u1 = self.up2(u2, x1)     # (B,64,256,64)
        u0 = self.up1(u1)         # (B,64,512,128)
        y  = self.final(u0)       # (B,6,512,128)
        return z, y

    def encode(self, x):  # x: (B,6,512,256)
        x1, s1 = self.down1(x)   # x1: (B,64,256,128)
        x2, s2 = self.down2(x1)  # x2: (B,128,128,64)
        x3, s3 = self.down3(x2)  # x3: (B,8,64,32)
        z = self.mid(x3)
        # 디코더에서 사용할 skip은 down의 "출력 해상도와 일치"하는 것들: x2, x1
        # (s1/s2는 원 해상도/전 단계 해상도이므로 사용하지 않음)
        return z, (x2, x1)

    def decode(self, z, skips):
        x2, x1 = skips
        u2 = self.up3(z, x2)     # -> (B,128,128,64)
        u1 = self.up2(u2, x1)    # -> (B,64,256,128)
        u0 = self.up1(u1)        # -> (B,64,512,256)
        y  = self.final(u0)      # -> (B,6,512,256)
        return y
# ----------------------------
# Train
# ----------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    spec = load_spec(args.spec)
    validate_spec(spec)

    ds = EmbWindowDataset(args.embed_dir, win=args.win, stride=args.stride, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(8, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    # ── 모델/옵티마이저 생성
    model = AEBackbone(c_in=6, c_lat=8).to(device)
    model = model.to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, fused=True)

    # ── 재개 상태
    start_epoch = 1
    global_step = 0

    if args.resume or args.resume_path:
        if args.resume_path is not None:
            ckpt_path = args.resume_path
        else:
            ckpt_path = _find_latest_ckpt(args.out_dir)

        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print(f"[resume] No checkpoint found (out_dir={args.out_dir}, resume_path={args.resume_path})")
        else:
            print(f"[resume] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)

            # 모델/옵티마이저 상태 복원
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])

            # 학습 진행 상태 복원
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            print(f"[resume] Resuming from epoch {start_epoch} (global_step={global_step})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            global_step += 1
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True, memory_format=torch.channels_last)

            accum = 2
            with torch.autocast("cuda", torch.bfloat16):
                _, yhat = model(x)
                mse = F.mse_loss(yhat, y)
                loss = (args.loss_scale * mse) / accum

            loss.backward()
            if (global_step % accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)

            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            pbar.set_postfix({
                "mse":  f"{mse.item():.6e}",
                "loss": f"{loss.item():.6e}",
                "avg":  f"{avg_loss:.6e}"
            })

        # ── 에폭 종료 로그
        epoch_avg = running_loss / max(1, len(dl))
        print(f"[epoch {epoch}] L_recon(epoch_avg)={epoch_avg:.6f}")

        # ── 에폭 단위 체크포인트 저장
        if epoch % args.ckpt_every_epoch == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"stage1_e{epoch:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    # 선택: 재현성 향상용 랜덤 상태 저장 (필요시 주석 해제)
                    # "rng_state": torch.get_rng_state(),
                    # "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    # "numpy_rng_state": np.random.get_state(),
                    # "args": vars(args),
                },
                ckpt_path
            )
            print(f"[ckpt] Saved: {ckpt_path}")

    print("stage1 끄으으으으으으읏")

def _find_latest_ckpt(out_dir: str) -> str | None:
    """
    out_dir 내 stage1_e*.pt 패턴에서 가장 최근(epoch 숫자 기준) 체크포인트를 찾음.
    없으면 수정시간(mtime) 기준으로도 대응함.
    """
    paths = sorted(glob.glob(os.path.join(out_dir, "stage1_e*.pt")))
    if not paths:
        # 과거 포맷 대비: stage1_*.pt 등 다른 파일도 고려 (mtime 기준)
        fallback = glob.glob(os.path.join(out_dir, "stage1_*.pt"))
        if not fallback:
            return None
        return max(fallback, key=os.path.getmtime)

    def _epoch_key(p: str) -> int:
        m = re.search(r"stage1_e(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths.sort(key=_epoch_key)
    return paths[-1] if paths else None



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--spec", type=str, required=True)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=50)

    ap.add_argument("--ckpt-every-epoch", type=int, default=16,
                    help="Save checkpoint every N epochs (epoch-based)")
    
    ap.add_argument("--out-dir", type=str, default="run_emb128/outputs_stage1")
    ap.add_argument("--epochs", type=int, default=256) 
    ap.add_argument("--loss-scale", type=float, default=1000.0,
                    help="loss가 처음부터 너무 작아서.. 업데이트에 한계가 있음. 스케일 조정인자")
    
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from the latest checkpoint in out-dir (or --resume-path)")
    ap.add_argument("--resume-path", type=str, default=None,
                    help="Explicit checkpoint path to resume from (overrides --resume)")


    
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

