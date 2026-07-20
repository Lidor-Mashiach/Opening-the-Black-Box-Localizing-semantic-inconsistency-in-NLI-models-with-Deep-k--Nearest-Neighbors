#!/bin/bash

# ---------------------------------------------------------------------------
# Submit the Optuna tuning sbatch jobs - BGU SLURM cluster.
# This script sits in the SAME folder as the .sbatch files and ONLY submits
# them - it does not copy or move anything.
# ---------------------------------------------------------------------------

BASE="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------------------------------------------------
# Active jobs - comment out any job you do NOT want to submit
# -----------------------------------------------------------------------
JOBS=(
    "$BASE/paraphrases_ANLI.sbatch"
    "$BASE/paraphrases_MNLI.sbatch"
    "$BASE/paraphrases_SNLI.sbatch"
 
)

echo "Submitting ${#JOBS[@]} job(s) from $BASE ..."
echo "----------------------------------------------------------------"

submitted=0
for JOB in "${JOBS[@]}"; do
    if [[ ! -f "$JOB" ]]; then
        echo "SKIP (not found): $JOB"
        continue
    fi
    JID=$(sbatch --parsable "$JOB")
    echo "Submitted job $JID: $(basename "$JOB")"
    submitted=$((submitted + 1))
done

echo "----------------------------------------------------------------"
echo "Done. $submitted job(s) submitted. Monitor with: squeue --me"