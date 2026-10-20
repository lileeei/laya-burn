#!/usr/bin/env python3
"""Compare two LAYA_DUMP_DIR outputs (rust vs reference)."""
import os, struct, sys

a_dir, b_dir = sys.argv[1], sys.argv[2]
names = sorted(set(f[:-4] for f in os.listdir(a_dir) if f.endswith(".f32")))
print(f"{'tensor':<18}{'shape':<16}{'max_abs':>10}{'rel_to_mag':>11}  verdict")
all_ok = True
for n in names:
    if not (os.path.exists(f"{b_dir}/{n}.f32") and os.path.exists(f"{a_dir}/{n}.f32")):
        print(f"{n:<18}MISSING on one side (skipped)"); continue
    av = struct.unpack(f"<{os.path.getsize(f'{a_dir}/{n}.f32')//4}f", open(f"{a_dir}/{n}.f32","rb").read())
    bv = struct.unpack(f"<{os.path.getsize(f'{b_dir}/{n}.f32')//4}f", open(f"{b_dir}/{n}.f32","rb").read())
    shape = open(f"{a_dir}/{n}.shape").read().strip()
    if len(av) != len(bv):
        print(f"{n:<18}{shape:<16}SIZE MISMATCH {len(av)} vs {len(bv)}"); all_ok = False; continue
    mabs = max((abs(x-y) for x, y in zip(av, bv)), default=0.0)
    mag = max((abs(x) for x in bv), default=1.0)
    rel = mabs / max(mag, 1e-9)
    ok = rel < 1e-3
    all_ok = all_ok and ok
    print(f"{n:<18}{shape:<16}{mabs:>10.2e}{rel:>11.2e}  {'OK' if ok else 'DIFF'}")
print("\nPARITY:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
