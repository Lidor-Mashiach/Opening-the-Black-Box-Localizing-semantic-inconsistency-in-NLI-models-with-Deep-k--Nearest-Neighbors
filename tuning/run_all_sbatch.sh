#!/bin/bash
### =====================================================================
### Deploy + submit the Optuna sbatch jobs - BGU SLURM cluster.
###
### NINE jobs, one per TRAINABLE model x dataset pair. The three large-model
### MNLI combinations use the labs' official published checkpoints and are
### NOT tuned, so they have no job here.
###
### Each job = ONE Optuna study -> configs/tuned/<MODEL>__<DATASET>.yaml
### (best hyper-parameters, NOT trained weights - the pipeline trains later).
###
### The .sbatch files carry ABSOLUTE paths, so this script copies them to your
### cluster scripts folder and submits them from there.
###
### Usage:
###   1. (once) check the two paths below match your cluster layout.
###   2. Comment out (put # before) any job you do NOT want to submit.
###   3. bash run_all_sbatch.sh
###   4. Monitor:  squeue --me        (cancel: scancel <job_id>)
###
### Hit the 7-day wall? Just submit that job again - run_tuning.py resumes the
### sqlite study from the next trial, nothing is lost.
### =====================================================================

REPO_SBATCH="$(cd "$(dirname "$0")" && pwd)/sbatch"
DEPLOY_DIR="/home/lidorma/sbatches_and_output_files/NLU-Scripts/sbatch-files"
OUTPUT_DIR="/home/lidorma/sbatches_and_output_files/NLU-Scripts/output-files"

JOBS=(
  ### ---------------- BERT-base (all three datasets are trained) ----------------
  "tune_BERT-base__SNLI.sbatch"
  "tune_BERT-base__MNLI.sbatch"
  "tune_BERT-base__ANLI.sbatch"
  ### ---------------- RoBERTa-large (MNLI uses the official checkpoint) ----------------
  "tune_RoBERTa-large__SNLI.sbatch"
  "tune_RoBERTa-large__ANLI.sbatch"
  ### ---------------- DeBERTa-large (MNLI uses the official checkpoint) ----------------
  "tune_DeBERTa-large__SNLI.sbatch"
  "tune_DeBERTa-large__ANLI.sbatch"
  ### ---------------- BART-large (MNLI uses the official checkpoint) ----------------
  "tune_BART-large__SNLI.sbatch"
  "tune_BART-large__ANLI.sbatch"
)

mkdir -p "$DEPLOY_DIR" "$OUTPUT_DIR"
echo "deploying sbatch files -> $DEPLOY_DIR"
echo "job logs -> $OUTPUT_DIR/<job-name>-<job-id>.out"
echo "----------------------------------------------------------------"

submitted=0
for job in "${JOBS[@]}"; do
  cp "$REPO_SBATCH/$job" "$DEPLOY_DIR/$job" || exit 1
  echo "submitting: $job"
  sbatch "$DEPLOY_DIR/$job" && submitted=$((submitted + 1))
done

echo "----------------------------------------------------------------"
echo "submitted $submitted job(s) (of 9 trainable combinations)."
echo "monitor:  squeue --me"
