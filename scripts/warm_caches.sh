#!/bin/sh
# Warm the CubeCL environment caches for laya-burn so first-run inference
# pays no JIT compilation. Produces target/environment/laya-warm.db.
#
# Ship that file with the application and activate it at runtime:
#   CUBECL_ENVIRONMENT=laya-warm ./laya-burn
# (The cache root defaults to <workspace>/target/environment; a bundled app
# can point it elsewhere with a cubecl.toml [environment] path entry.)
set -e
cd "$(dirname "$0")/.."

export CUBECL_ENVIRONMENT=laya-warm
echo "warming shape buckets into environment 'laya-warm' ..."

for n in 1 2 4 8; do
  LAYA_BATCH=$n ./target/release/laya-burn > /dev/null 2>&1
  echo "  batch $n warmed"
done
LAYA_LONG=1 ./target/release/laya-burn > /dev/null 2>&1
echo "  long-sequence warmed"

echo "done:"; ls -la target/environment/ | grep laya-warm || true
