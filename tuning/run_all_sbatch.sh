#!/bin/bash
### =====================================================================
### Main runner for the Optuna sbatch jobs - BGU SLURM cluster.
###
### Usage:
###   1. Comment out (put # before) every job you do NOT want to submit.
###   2. bash run_all_sbatch.sh
###   3. Monitor with:  squeue --me      (cancel with: scancel <job_id>)
###
### Each job = one Optuna study for one model x dataset pair, up to ~7 days,
### 1 GPU, fully resumable (resubmit after a crash and it continues).
### Before the first submission, edit the CHANGE-ME lines inside the
### .sbatch files (conda env name + repo path).
### =====================================================================
cd "$(dirname "$0")"

JOBS=(
  ### ---------------- BERT-base ----------------
  "sbatch/tune_BERT-base__SNLI.sbatch"
  "sbatch/tune_BERT-base__MNLI.sbatch"
  "sbatch/tune_BERT-base__ANLI.sbatch"
  ### ---------------- RoBERTa-large ----------------
  "sbatch/tune_RoBERTa-large__SNLI.sbatch"
  "sbatch/tune_RoBERTa-large__MNLI.sbatch"
  "sbatch/tune_RoBERTa-large__ANLI.sbatch"
  ### ---------------- DeBERTa-large ----------------
  "sbatch/tune_DeBERTa-large__SNLI.sbatch"
  "sbatch/tune_DeBERTa-large__MNLI.sbatch"
  "sbatch/tune_DeBERTa-large__ANLI.sbatch"
  ### ---------------- BART-large ----------------
  "sbatch/tune_BART-large__SNLI.sbatch"
  "sbatch/tune_BART-large__MNLI.sbatch"
  "sbatch/tune_BART-large__ANLI.sbatch"
)

submitted=0
for job in "${JOBS[@]}"; do
  echo "submitting: $job"
  sbatch "$job" && submitted=$((submitted + 1))
done
echo "----------------------------------------------------------------"
echo "submitted $submitted job(s). Check the queue with:  squeue --me"
