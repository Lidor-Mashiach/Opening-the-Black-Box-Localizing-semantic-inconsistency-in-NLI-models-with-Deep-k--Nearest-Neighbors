#!/bin/bash
### Paraphrase bank generation: ANLI ONLY  -  "Opening the Black Box" (NLI-DkNN)
### Builds Datasets/ANLI-Adversarial_NLI/paraphrases/paraphrase_bank.csv from the
### ANLI train split. Independent of the MNLI/ANLI jobs - run all three in
### parallel to build every bank at once (the per-dataset reset only touches
### ANLI, so concurrent jobs never collide).
### Submit:  sbatch paraphrases_ANLI.sbatch
### Resumable: an existing bank is SKIPPED; delete the bank + its stats.json to
### force a clean rebuild.

#SBATCH --partition main
#SBATCH --time 2-23:59:00
#SBATCH --job-name paraphrases_ANLI
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
python -u setup-files/Paraphrase-Generator/generate_paraphrases.py --datasets ANLI
