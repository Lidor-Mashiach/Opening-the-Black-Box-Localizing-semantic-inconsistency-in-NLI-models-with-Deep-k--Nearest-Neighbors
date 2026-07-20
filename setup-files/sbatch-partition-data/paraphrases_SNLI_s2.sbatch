#!/bin/bash
### Paraphrase bank generation: SNLI - SHARD 3/3  -  "Opening the Black Box" (NLI-DkNN)
### One of THREE parallel jobs that split SNLI's train hypotheses into
### disjoint slices. This job writes ONLY its own part file:
###     Datasets/*/paraphrases/shards/paraphrase_bank.part002of003.csv
### Parts never collide (disjoint slices, per-process atomic writes) and shard
### jobs never reset derived data - so all 9 shard jobs (3 datasets x 3
### shards) run concurrently with zero conflicts.
### AFTER all 3 SNLI shards finish, assemble the final bank ONCE:
###     python setup-files/Paraphrase-Generator/merge_shards.py --datasets SNLI
### Submit:  sbatch paraphrases_SNLI_s2.sbatch     (or all 9: bash run_all_shard_generators.sh)
### Resumable: an existing part is SKIPPED; a job killed mid-run leaves NO part
### (atomic write), so simply resubmit and the shard redoes itself cleanly.

#SBATCH --partition main
#SBATCH --time 2-23:59:00
#SBATCH --job-name paraphrases_SNLI_s2
#SBATCH --output /home/lidorma/sbatches_and_output_files/NLU-Scripts/output-files/%x.out
#SBATCH --mail-user=lidorma@post.bgu.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=58G

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
nvidia-smi -L

module load anaconda
source activate nlu_env

cd /home/lidorma/projects/NLU-Project
python -u setup-files/Paraphrase-Generator/generate_paraphrases.py --datasets SNLI --num-shards 3 --shard-index 2
