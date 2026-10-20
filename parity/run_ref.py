#!/usr/bin/env python3
"""Parity reference: run the official Laya implementation and dump intermediates
matching the Rust side (LAYA_DUMP_DIR layout) for row 0 of the batch."""
import json, os, struct, sys

import torch

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "model")
OUT = os.environ.get("LAYA_DUMP_DIR", "parity/ref_out")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, MODEL_DIR)
from rl_common import QTYPES, build_model, build_sequence, collate_items, render_options  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def dump(name, t):
    v = t.detach().float().cpu().reshape(-1).contiguous()
    with open(f"{OUT}/{name}.f32", "wb") as f:
        f.write(struct.pack(f"<{v.numel()}f", *v.tolist()))
    with open(f"{OUT}/{name}.shape", "w") as f:
        f.write(repr(list(t.shape)))


def main():
    with open(f"{MODEL_DIR}/rl_agent_config.json") as fh:
        cfg = json.load(fh)
    tok = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/tokenizer")
    model = build_model(cfg, encoder_dir=f"{MODEL_DIR}/typed-decisions/encoder")
    model.load_state_dict(load_file(f"{MODEL_DIR}/model.safetensors"), strict=True)
    model.eval()

    # same questions as src/main.rs, same order
    state = "The user is deciding whether to run a 421M non-autoregressive decision model in Rust. Facts: Burn is a Rust tensor framework; a checkpoint is available."
    qs = [
        ("choice", "Which option is the largest number?", [str(n) for n in range(1, 13)]),
        ("noul", "The word 'apple' contains the letter 'a'.", None),
        ("choice", "Which runtime should host the model for CPU inference in Rust?",
         ["burn", "candle", "onnxruntime", "libtorch"]),
        ("noul", "The checkpoint runs natively in Rust without any Python dependency.", None),
        ("score", "Rate the usefulness of a zero-token decision model for routing.",
         ["useless", "marginal", "useful", "very useful", "essential"]),
    ]
    items = []
    for t, ins, crit in qs:
        if t == "choice":
            crit_d = {c: None for c in crit}
        else:
            crit_d = crit
        q = {"t": t, "ins": ins, "crit": crit_d}
        seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t],
                      "target": [0.0] * len(markers), "label": -1,
                      "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
    b = collate_items([items], tok.pad_token_id)
    with open(f"{OUT}/all_rows.txt", "w") as f:
        f.write(chr(10).join(",".join(str(x) for x in row.tolist()) for row in b["input_ids"]))
    with open(f"{OUT}/input_ids.txt", "w") as f:
        f.write(",".join(str(x) for x in b["input_ids"][0].tolist()))

    ids = b["input_ids"].long()
    amask = b["attention_mask"].long()

    # --- hooks mirroring the Rust dump points ---
    enc = model.encoder
    dumps = {}

    def hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            dumps.setdefault(name, t.detach())
        return fn

    handles = [
        enc.layers[0].attn.register_forward_hook(hook("l0_attn_module")),
        enc.layers[0].attn.Wqkv.register_forward_hook(hook("l0_qkv_out")),
        enc.embeddings.register_forward_hook(hook("emb_out")),
        enc.layers[0].register_forward_hook(hook("enc_layer_0")),
        enc.layers[1].register_forward_hook(hook("enc_layer_1")),
        enc.layers[len(enc.layers) - 1].register_forward_hook(hook("enc_layer_%d" % (len(enc.layers) - 1))),
        enc.final_norm.register_forward_hook(hook("encoder_out")),
        model.head.layers[len(model.head.layers) - 1].register_forward_hook(hook("head_out")),
    ]
    with torch.no_grad():
        logits, act = model(ids, amask, b["marker_pos"].long(), b["marker_mask"].bool(),
                            torch.tensor([it["qtype"] for it in items]))
    for h in handles:
        h.remove()

    # layer0 q after rope, computed per HF semantics
    attn0 = enc.layers[0].attn
    qkv = dumps["l0_qkv_out"]
    bs, sl = qkv.shape[0], qkv.shape[1]
    qkv5 = qkv.view(bs, -1, 3, attn0.num_heads, attn0.head_dim)
    q = qkv5.transpose(3, 1).unbind(dim=2)[0]
    pos = torch.arange(sl).unsqueeze(0)
    cos, sin = attn0.rotary_emb(q, position_ids=pos)
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    rot = torch.cat((-q2, q1), dim=-1)
    q_rope = (q * cos) + (rot * sin)
    dumps["l0_q_after_rope"] = q_rope[0]

    for name, t in dumps.items():
        row = t[0]
        dump(name, row)
    dump("logits", logits[0])
    dump("act_logits", act[0])
    print("reference dumps written to", OUT)


if __name__ == "__main__":
    main()

