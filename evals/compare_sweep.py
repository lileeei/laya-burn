#!/usr/bin/env python3
import json
import os
import statistics

ours = json.load(open(os.environ.get('LAYA_SWEEP_OURS', '/tmp/sweep51_ours.json')))
ref = json.load(open('/tmp/laya-official/research/results/cpu_51_language_sweep.json'))
ref_pl = ref['part_a']['by_model']['english']['per_language']

rows = []
exact = 0
for lang in sorted(ours):
    if lang not in ref_pl:
        continue
    o = ours[lang]['accuracy']
    r = ref_pl[lang]['accuracy']
    rows.append((lang, o, r, o - r))
    if abs(o - r) < 1e-9:
        exact += 1

print('lang   ours    ref     delta')
print('-' * 34)
for lang, o, r, d in rows:
    print(f'{lang:6} {o:.4f}  {r:.4f}  {d:+.4f}')
print('-' * 34)
m_o = statistics.mean(o for _, o, _, _ in rows)
m_r = statistics.mean(r for _, _, r, _ in rows)
print(f'macro  {m_o:.4f}  {m_r:.4f}  {m_o - m_r:+.4f}   (n={len(rows)}, exact={exact})')
