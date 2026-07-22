#!/bin/bash

# ---------------------------------------------------------------------------
# Submit the 9 SHARDED paraphrase-generation jobs - BGU SLURM cluster.
# 3 datasets x 3 disjoint shards = 9 jobs that run fully in parallel:
#   * each writes ONLY its own paraphrase_bank.partIIIofNNN.csv (never collide)
#   * shard jobs never reset derived data (no job wipes another's work)
#   * a killed job leaves NO part behind (atomic write) - just resubmit it
#
# This script sits in the SAME folder as the .sbatch files and ONLY submits
# them - it does not copy or move anything.
#
# AFTER all 9 finish (squeue --me is empty), assemble the 3 final banks:
#     python setup-files/Paraphrase-Generator/merge_shards.py
# (or per dataset: merge_shards.py --datasets SNLI). merge_shards.py writes
# each bank to the canonical path Phase A reads, then resets the derived
# pipeline once. It refuses to build an incomplete bank if a shard is missing.
# ---------------------------------------------------------------------------

BASE="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------------------------------------------------
# Active jobs - comment out any shard you do NOT want to submit
# -----------------------------------------------------------------------
JOBS=(
    "$BASE/paraphrases_SNLI_s0.sbatch"
    "$BASE/paraphrases_SNLI_s1.sbatch"
    "$BASE/paraphrases_SNLI_s2.sbatch"
    "$BASE/paraphrases_MNLI_s0.sbatch"
    "$BASE/paraphrases_MNLI_s1.sbatch"
    "$BASE/paraphrases_MNLI_s2.sbatch"
    "$BASE/paraphrases_ANLI_s0.sbatch"
    "$BASE/paraphrases_ANLI_s1.sbatch"
    "$BASE/paraphrases_ANLI_s2.sbatch"
)

echo "Submitting ${#JOBS[@]} shard job(s) from $BASE ..."
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
echo "Done. $submitted shard job(s) submitted. Monitor with: squeue --me"
echo ""
echo "When ALL shards finish, assemble the banks:"
echo "    python setup-files/Paraphrase-Generator/merge_shards.py"