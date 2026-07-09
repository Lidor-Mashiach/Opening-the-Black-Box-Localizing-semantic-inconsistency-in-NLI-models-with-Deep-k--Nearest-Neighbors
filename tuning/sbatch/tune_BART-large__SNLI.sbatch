#!/bin/bash
### Optuna tuning job: BART-large x SNLI - "Opening the Black Box" (NLI-DkNN)
### BGU SLURM cluster (slurm.bgu.ac.il). Submit from the tuning/ directory
### (run_all_sbatch.sh does this for you):   sbatch sbatch/tune_BART-large__SNLI.sbatch
### Resumable: if the job crashes / is pre-empted, just resubmit - the Optuna
### sqlite study continues from the next trial.

#SBATCH --partition main            ### regular accounts: main only
#SBATCH --time 6-23:59:00           ### just under the 7-day cluster maximum
#SBATCH --job-name tune_BART-large__SNLI
#SBATCH --output outputs/%x-%J.out  ### -> tuning/outputs/ (submit from tuning/)
#SBATCH --gpus=1                    ### 1 GPU per job (cluster policy)
#SBATCH --mem=32G                   ### default is 24G/GPU; 60G is the hard cap
##SBATCH --constraint=rtx_3090      ### optional: pin a GPU type (rtx_3090 / rtx_4090 / ...)
##SBATCH --mail-user=your_user@post.bgu.ac.il   ### CHANGE-ME + uncomment for email updates
##SBATCH --mail-type=BEGIN,END,FAIL

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
nvidia-smi -L

export PYTHONUNBUFFERED=TRUE        ### live logs in the .out file

module load anaconda
source activate nli_dknn            ### CHANGE-ME: your conda env (pip install -r requirements.txt)

REPO="$HOME/projects/Opening-The-Black-Box"   ### CHANGE-ME: repo path on the cluster
cd "$REPO"

python -u tuning/run_tuning.py --model BART-large --dataset SNLI
