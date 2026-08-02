#!/usr/bin/env bash
#
# Runs every project end to end in --quick mode: a handful of batches through
# train.py, then evaluate.py. It checks that the pipelines work, not that the
# models are any good -- the numbers a quick run produces are meaningless.
#
#   ./verify.sh              # all projects
#   ./verify.sh 08-ddpm      # just one
#
# Expect it to take several minutes on a CPU. The diffusion projects dominate,
# because sampling is a sequential loop over timesteps.

set -u
cd "$(dirname "$0")" || exit 1

ONLY="${1:-}"
FAILED=0
PASSED=0
export PYTHONUNBUFFERED=1

check() {
  local dir="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$dir" ]; then return; fi
  printf '%-22s %s ... ' "$dir" "$1"
  if (cd "$dir" && "$@" >/tmp/verify-$$.log 2>&1); then
    echo "ok"
    PASSED=$((PASSED + 1))
  else
    echo "FAILED"
    sed 's/^/    /' /tmp/verify-$$.log | tail -15
    FAILED=$((FAILED + 1))
  fi
  rm -f /tmp/verify-$$.log
}

Q="--quick --num-workers 0"

# shellcheck disable=SC2086
{
check 01-vae              python3 train.py $Q
check 01-vae              python3 evaluate.py $Q
check 02-vq-vae           python3 train.py $Q --dataset shapes
check 02-vq-vae           python3 evaluate.py $Q
check 03-dcgan            python3 train.py $Q
check 03-dcgan            python3 evaluate.py $Q
check 04-conditional-gan  python3 train.py $Q
check 04-conditional-gan  python3 evaluate.py $Q
check 05-pix2pix          python3 train.py $Q --dataset shapes
check 05-pix2pix          python3 evaluate.py $Q
check 06-cyclegan         python3 train.py $Q --task datasets --domain-a mnist --domain-b fashion-mnist
check 06-cyclegan         python3 evaluate.py $Q
check 07-realnvp          python3 train.py $Q
check 07-realnvp          python3 evaluate.py $Q
check 08-ddpm             python3 train.py $Q
check 08-ddpm             python3 evaluate.py $Q
check 09-ddim             python3 train.py --num-workers 0
check 09-ddim             python3 evaluate.py $Q
check 10-latent-diffusion python3 train.py $Q
check 10-latent-diffusion python3 evaluate.py $Q
check 11-vit              python3 train.py $Q --dataset shapes
check 11-vit              python3 train.py $Q --dataset shapes --model resnet18
check 11-vit              python3 evaluate.py $Q
}

echo
echo "passed: $PASSED   failed: $FAILED"
exit $((FAILED > 0))
