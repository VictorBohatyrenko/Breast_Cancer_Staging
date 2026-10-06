#!/usr/bin/env bash
# DTFD-ASMIL v2 на UNI2h backbone, 10 сідів кожен, C17 + BRACS.
# GPU0: C17 (10 seeds), GPU1: BRACS (10 seeds) -- паралельно, ~одночасно готові.
#
# tier1_weight=0.1, n_token=M=8 -- ті самі гіперпараметри, що й у ViT-S/DINO
# DTFD-ASMIL v2 прогонах (для чесного порівняння backbone-ефекту окремо від
# зміни архітектурних налаштувань).

set -e
mkdir -p logs ckpt

TIER1_WEIGHT=0.1
N_TOKEN=8

run_dtfd_v2_uni2h () {
    local dataset_config=$1
    local tag=$2
    local seed=$3
    python Step3_WSI_classification_DTFD_ASMIL_v2.py \
        --config "$dataset_config" --pretrain UNI2h \
        --n_token "$N_TOKEN" --tier1_weight "$TIER1_WEIGHT" --seed "$seed" \
        --ckpt_dir "ckpt/${tag}_dtfd_v2_uni2h_seed${seed}" --wandb_mode offline \
        > "logs/${tag}_dtfd_v2_uni2h_seed${seed}.log" 2>&1
}

echo "GPU0: C17 DTFD-ASMIL v2 / UNI2h, 10 seeds"
(
  export CUDA_VISIBLE_DEVICES=0
  for s in 1 2 3 4 5 6 7 8 9 10; do
      run_dtfd_v2_uni2h "config/camelyon17_uni2h_config.yml" "c17" "$s"
      echo "  c17_dtfd_v2_uni2h seed${s} готовий"
  done
) &
PID_GPU0=$!

echo "GPU1: BRACS DTFD-ASMIL v2 / UNI2h, 10 seeds"
(
  export CUDA_VISIBLE_DEVICES=1
  for s in 1 2 3 4 5 6 7 8 9 10; do
      run_dtfd_v2_uni2h "config/bracs_uni2h_config.yml" "bracs" "$s"
      echo "  bracs_dtfd_v2_uni2h seed${s} готовий"
  done
) &
PID_GPU1=$!

echo "Черги запущені (GPU0 PID=$PID_GPU0, GPU1 PID=$PID_GPU1). Чекаю завершення..."
wait $PID_GPU0
wait $PID_GPU1

echo ""
echo "Усі 20 запусків завершено."
echo ""
echo "Витягнути фінальні результати:"
echo 'for tag in c17 bracs; do'
echo '  for seed in 1 2 3 4 5 6 7 8 9 10; do'
echo '    echo "=== ${tag} seed${seed} ==="'
echo '    grep -A 3 "FINAL EPOCH" logs/${tag}_dtfd_v2_uni2h_seed${seed}.log'
echo '  done'
echo 'done'
