#!/usr/bin/env bash
#
# Runs every project end to end in --quick mode: a handful of batches through
# train.py, then evaluate.py. It checks that the pipelines work, not that the
# models are any good -- the numbers a quick run produces are meaningless.
#
#   ./verify.sh              # all projects
#   ./verify.sh 13_ddpm      # just one
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

# The metric code is checked first: if FID or precision/recall are wrong, every
# number the projects report downstream is meaningless.
printf '%-22s %s ... ' "common" "metric self-test"
if python3 -m common.selftest >/tmp/verify-$$.log 2>&1; then
  echo "ok"; PASSED=$((PASSED + 1))
else
  echo "FAILED"; sed 's/^/    /' /tmp/verify-$$.log | tail -15; FAILED=$((FAILED + 1))
fi
rm -f /tmp/verify-$$.log

Q="--quick --num-workers 0"

# shellcheck disable=SC2086
{
check 06_vae              python3 train.py $Q
check 06_vae              python3 evaluate.py $Q
check 07_vq_vae           python3 train.py $Q --dataset shapes
check 07_vq_vae           python3 evaluate.py $Q
check 08_dcgan            python3 train.py $Q
check 08_dcgan            python3 evaluate.py $Q
check 09_conditional_gan  python3 train.py $Q
check 09_conditional_gan  python3 evaluate.py $Q
check 10_pix2pix          python3 train.py $Q --dataset shapes
check 10_pix2pix          python3 evaluate.py $Q
check 11_cyclegan         python3 train.py $Q --task datasets --domain-a mnist --domain-b fashion-mnist
check 11_cyclegan         python3 evaluate.py $Q
check 12_realnvp          python3 train.py $Q
check 12_realnvp          python3 evaluate.py $Q
check 13_ddpm             python3 train.py $Q
check 13_ddpm             python3 evaluate.py $Q
check 14_ddim             python3 train.py --num-workers 0
check 14_ddim             python3 evaluate.py $Q
check 15_latent_diffusion python3 train.py $Q
check 15_latent_diffusion python3 evaluate.py $Q
check 16_vit              python3 train.py $Q --dataset shapes
check 16_vit              python3 train.py $Q --dataset shapes --model resnet18
check 16_vit              python3 evaluate.py $Q
}

echo
echo "passed: $PASSED   failed: $FAILED"
exit $((FAILED > 0))
