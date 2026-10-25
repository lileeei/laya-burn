#!/bin/bash
# Reproduce the es_phone fine-tuning experiment (evals/es_phone_finetune_comparison.json)
# Prerequisites: LAYA_OFFICIAL set to the official repo clone (for build_train.py and eval)
#               model-multilingual/ in the repo root

set -e
OFFICIAL=${LAYA_OFFICIAL:-/tmp/laya-official}

echo "=== 1. Generate training data (4740 items) ==="
cd "$OFFICIAL/research/benchmarks/es_phone_turns/train"
python3 build_train.py --out /tmp/es_train.jsonl
cd -

echo "=== 2. Convert soft targets to hard labels ==="
python3 -c "
import json
rows = [json.loads(l) for l in open('/tmp/es_train.jsonl') if l.strip()]
out = [{'state': r['state'], 'questions': r['questions'], 'expected': {q: g['label'] for q, g in r.get('gold', {}).items()}} for r in rows]
with open('/tmp/es_train_hard.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'wrote {len(out)} hard-label items')
"

echo "=== 3. Fine-tune full (2 epochs) ==="
rm -rf ft-es-full
./target/release/laya-burn finetune /tmp/es_train_hard.jsonl model-multilingual ft-es-full 2 4

echo "=== 4. Fine-tune freeze_encoder (2 epochs) ==="
rm -rf ft-es-fz
LAYA_FREEZE_ENCODER=1 ./target/release/laya-burn finetune /tmp/es_train_hard.jsonl model-multilingual ft-es-fz 2 4

echo "=== 5. Evaluate all three ==="
for variant in base full fz; do
  echo "--- $variant ---"
  pkill -f laya-serve 2>/dev/null; sleep 1
  case $variant in
    base) (LAYA_MODEL=multilingual ./target/release/laya-serve 8450 > /tmp/sv_$variant.log 2>&1 &); dir="" ;;
    *) mkdir -p /tmp/eval_$variant; ln -sfn "$(pwd)/ft-es-$variant" /tmp/eval_$variant/model-multilingual
       (cd /tmp/eval_$variant && LAYA_MODEL=multilingual "$(pwd)/target/release/laya-serve" 8450 > /tmp/sv_$variant.log 2>&1 &); dir="/tmp/eval_$variant" ;;
  esac
  sleep 15
  curl -s localhost:8450/v1/decide -H 'Content-Type: application/json' -d '{"state":"w","questions":{"q":{"type":"noul","instructions":"?"}}}' > /dev/null
  python3 evals/run_es_phone.py http://127.0.0.1:8450 2>&1 | grep -E "cases:|independent:"
  pkill -f laya-serve 2>/dev/null
done
echo "=== done ==="
