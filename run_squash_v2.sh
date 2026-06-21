#!/usr/bin/env bash
# Current-code SQUASH retrain = the paper's literal training protocol
# ("resize all images to 640x640" + patch = 25% of bbox AREA), as the zero-confound
# counterpart to output_v1 (which used letterbox). Same code / hyperparameters as
# output_v1; ONLY --resize changes (letterbox -> squash).
#
# Chains: AdvPatch -> CAPGen-T1(Bc1) -> CAPGen-T2(Bc2) -> linear PAC-P (+ softmax R)
#         -> run_table1 (squash, frac0.25, conf0.001, --p_method linear).
set -u
PY="F:/Anaconda/envs/capgen/python.exe"
OUT="output_v2_squash"
DS="./INRIAPerson"
mkdir -p "$OUT"
LOG="$OUT/train_all.log"

echo "==== TRAIN ALL (squash, frac=0.25, center, 200 epochs) started: $(date) ====" | tee "$LOG"

echo "---- [1/3] AdvPatch -> $OUT/advpatch ----" | tee -a "$LOG"
"$PY" advpatch_trainer.py --dataset_dir "$DS" \
  --num_iterations 200 --batch_size 8 --lr 0.03 \
  --resize squash --patch_frac 0.25 \
  --output_dir "$OUT/advpatch" >> "$LOG" 2>&1
echo "ADVPATCH_EXIT=$?" | tee -a "$LOG"

echo "---- [2/3] CAPGen-T1 (Bc1) -> $OUT/capgen_t1 ----" | tee -a "$LOG"
"$PY" main.py --mode train --dataset_dir "$DS" --base_colors bc1 \
  --num_iterations 200 --resize squash --patch_frac 0.25 \
  --output_dir "$OUT/capgen_t1" >> "$LOG" 2>&1
echo "T1_EXIT=$?" | tee -a "$LOG"

echo "---- [3/3] CAPGen-T2 (Bc2) -> $OUT/capgen_t2 ----" | tee -a "$LOG"
"$PY" main.py --mode train --dataset_dir "$DS" --base_colors bc2 \
  --num_iterations 200 --resize squash --patch_frac 0.25 \
  --output_dir "$OUT/capgen_t2" >> "$LOG" 2>&1
echo "T2_EXIT=$?" | tee -a "$LOG"

echo "---- [P] linear PAC-P + softmax R from new squash AdvPatch ----" | tee -a "$LOG"
"$PY" make_capgen_p.py --method advpatch-rgb-linear \
  --advpatch "$OUT/advpatch/best_advpatch.pt" \
  --out_dir "$OUT/capgen_p_linear" >> "$LOG" 2>&1
echo "PLINEAR_EXIT=$?" | tee -a "$LOG"
"$PY" make_capgen_p.py --method advpatch-rgb \
  --advpatch "$OUT/advpatch/best_advpatch.pt" \
  --out_dir "$OUT/capgen_p" >> "$LOG" 2>&1
echo "PSOFT_EXIT=$?" | tee -a "$LOG"

echo "---- [TABLE] squash conf0.001 linear ----" | tee -a "$LOG"
"$PY" run_table1.py \
  --advpatch "$OUT/advpatch/best_advpatch.pt" \
  --t1 "$OUT/capgen_t1/best_color_prob.pt" \
  --t2 "$OUT/capgen_t2/best_color_prob.pt" \
  --cp_dir "$OUT/capgen_p" \
  --p_dir "$OUT/capgen_p_linear" --p_method linear \
  --dataset_dir "$DS" \
  --resize squash --patch_frac 0.25 --conf 0.001 \
  --out_json "$OUT/table1_squash_f025_center_linear_conf0001.json" >> "$LOG" 2>&1
echo "TABLE_EXIT=$?" | tee -a "$LOG"

echo "==== DONE: $(date) ====" | tee -a "$LOG"
