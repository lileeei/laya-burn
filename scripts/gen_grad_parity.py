#!/usr/bin/env python3
"""Generate a fixed batch, run the OFFICIAL rlcd_loss on it (eps injected),
and dump loss + d(loss)/d(logits) for comparison against the Rust core."""
import argparse
import json, sys
import numpy as np
import torch

OUT = "tests/fixtures/grad_parity"

sys.path.insert(0, "/tmp/laya-official")
from laya.common import proper_reward  # official reward, verbatim

torch.set_default_dtype(torch.float32)
R, C, S = 4, 6, 4          # rows, cols, samples
SIGMA, W_SPH, W_RPS = 0.4, 0.75, 1.0

rng = np.random.default_rng(42)
# logits: modest magnitudes like a mid-training model
logits_np = rng.normal(0.0, 1.5, size=(R, C)).astype(np.float32)
# targets: two one-hot rows, one soft row, one ordinal
target_np = np.zeros((R, C), dtype=np.float32)
target_np[0, 2] = 1.0
target_np[1, 5] = 1.0
target_np[2, 1] = 0.6; target_np[2, 2] = 0.4
target_np[3, :3] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
mask_np = np.ones((R, C), dtype=np.float32)
qtype = torch.tensor([0, 0, 0, 1])  # last row ordinal (score)
# eps: gaussian, projected per row like the official
eps_np = rng.normal(0.0, SIGMA, size=(S, R, C)).astype(np.float32)
fmask = torch.tensor(mask_np)
eps = torch.tensor(eps_np)
k = fmask.sum(-1, keepdim=True)
eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask

logits = torch.tensor(logits_np, requires_grad=True)
target = torch.tensor(target_np)
_MASKED = torch.finfo(torch.float32).min

def official_rlcd(logits, target, mask, qtype, sigma, samples, eps, w_sph=0.75, w_rps=1.0):
    """train.py rlcd_loss with eps injected instead of torch.randn."""
    logp = torch.log_softmax(logits.masked_fill(mask == 0, _MASKED), -1)
    ce = -(target * logp * mask).sum(-1).mean()
    fmask = mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(fmask == 0, _MASKED), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, fmask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp_g = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    return -(adv * logp_g).mean() + ce, r, adv

loss, r, adv = official_rlcd(logits, target, fmask, qtype, SIGMA, S, eps)
loss.backward()
# decompose
logp_ce = torch.log_softmax(logits.detach().masked_fill(fmask == 0, _MASKED), -1)
ce_val = float(-(target * logp_ce * fmask).sum(-1).mean())
z_d = logits.detach().unsqueeze(0) + eps
logp_g = -(((z_d - logits.detach().unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * SIGMA ** 2)
pg_val = float(-(adv * logp_g).mean())
print("py ce:", ce_val, " py pg:", pg_val)

def dump(name, arr):
    with open(f"{OUT}/{name}.f32", "wb") as f:
        f.write(arr.astype(np.float32).tobytes())

dump("logits", logits_np); dump("target", target_np); dump("mask", mask_np)
dump("eps", eps.detach().numpy().transpose(1, 0, 2).reshape(R * S, C))  # row-major [r*S+s]
dump("py_grad", logits.grad.detach().numpy())
json.dump({"rows": R, "cols": C, "samples": S, "sigma": SIGMA, "w_sph": W_SPH,
           "w_rps": W_RPS, "qtype": qtype.tolist(),
           "py_loss": float(loss), "py_ce": ce_val, "py_pg": pg_val, "py_rewards": r.tolist(), "py_adv": adv.tolist()},
          open(f"{OUT}/meta.json", "w"))
print("py loss:", float(loss))
print("rewards[0]:", [round(x, 4) for x in r[:, 0].tolist()])
print("adv[0]:   ", [round(x, 4) for x in adv[:, 0].tolist()])
