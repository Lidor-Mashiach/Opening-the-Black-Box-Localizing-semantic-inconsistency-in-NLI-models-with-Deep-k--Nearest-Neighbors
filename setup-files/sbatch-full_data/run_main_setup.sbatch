#!/bin/bash
### One-time environment setup: main-setup.py  -  "Opening the Black Box" (NLI-DkNN)
### Runs the 3 setup STEPS in order: environment -> datasets -> paraphrases.
### The paraphrase step is the long one (it paraphrases EVERY train hypothesis
### in all three datasets), so this runs as a batch job - submit it, close your
### laptop, and the paraphrase banks keep building on the node.
### Submit:  sbatch run_main_setup.sbatch
### Resumable-ish: each step skips work already done (env keeps a good torch,
### datasets skip splits already downloaded, paraphrases skip banks that exist),
### so if it is interrupted, submitting again resumes from where it stopped.

#SBATCH --partition main
#SBATCH --time 2-23:59:00
#SBATCH --job-name run_main_setup
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
python -u main-setup.py
