#!/bin/bash
### Optuna hyper-parameter SEARCH: BART-large x ANLI  -  "Opening the Black Box" (NLI-DkNN)
### Product: configs/tuned/BART-large__ANLI.yaml = the best hyper-parameters
### for this combination (NOT trained weights - the pipeline trains later).
### Submit:  sbatch tune_BART-large__ANLI.sbatch        (or all of them: bash run_all_sbatch.sh)
### Resumable: crashed / hit the wall? Just submit again - the code (run_tuning.py)
### resumes the sqlite study from the next trial.

#SBATCH --partition main
#SBATCH --time 6-23:59:00
#SBATCH --job-name tune_BART-large__ANLI
#SBATCH --output /home/lidorma/sbatches_and_output_files/NLU-Scripts/output-files/%x.out
#SBATCH --mail-user=lidorma@post.bgu.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=32G

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
nvidia-smi -L

module load anaconda
source activate nlu_env

cd /home/lidorma/projects/NLU-Project
python -u tuning/run_tuning.py --model BART-large --dataset ANLI
