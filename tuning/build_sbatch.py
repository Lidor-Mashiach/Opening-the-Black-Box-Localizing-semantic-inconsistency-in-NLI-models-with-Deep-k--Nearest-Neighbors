"""Regenerate the Optuna sbatch files from ONE template.

Only the NINE trainable combinations get a job: the three large-model MNLI
combinations use the labs' OFFICIAL published checkpoints and never tune.

Edit the constants below (cluster paths, conda env, mail, per-model GPU) and
run:  python tuning/build_sbatch.py
Every .sbatch file is rewritten consistently - beats editing files by hand
and keeps every resource decision documented in one place
(see tuning/README.md).
"""
from pathlib import Path

SBATCH_DIR = Path(__file__).resolve().parent / "sbatch"
OUT_DIR = "/home/lidorma/sbatches_and_output_files/NLU-Scripts/output-files"
REPO = "/home/lidorma/projects/NLU-Project"
ENV = "nlu_env"
MAIL = "lidorma@post.bgu.ac.il"

# The three MNLI combinations below use official published checkpoints and are
# NOT tuned - so they get no sbatch file.
OFFICIAL_MNLI = {"RoBERTa-large", "DeBERTa-large", "BART-large"}
MODELS = ["BERT-base", "RoBERTa-large", "DeBERTa-large", "BART-large"]
DATASETS = ["SNLI", "MNLI", "ANLI"]

# Per-model GPU request. A large-model trial peaks at ~9-12GB VRAM, so it needs
# a 24GB-class card; BERT-base needs < 6GB and takes any GPU. Requesting the
# type via --gpus (gpus=<type>:1) is the BGU-documented way; --constraint is
# not needed, and asking for a GPU already allocates 4-6 CPUs automatically.
GPU_LARGE = "rtx_3090:1"     # 24GB; swap to rtx_4090:1 / rtx_6000:1 if preferred
GPU_BASE = "1"               # any available GPU

TEMPLATE = """#!/bin/bash
### Optuna hyper-parameter SEARCH: {model} x {dataset}  -  "Opening the Black Box" (NLI-DkNN)
### Product: configs/tuned/{model}__{dataset}.yaml = the best hyper-parameters
### for this combination (NOT trained weights - the pipeline trains later).
### Submit:  sbatch {fname}        (or all of them: bash run_all_sbatch.sh)
### Resumable: crashed / hit the wall? Just submit again - the code (run_tuning.py)
### resumes the sqlite study from the next trial.

#SBATCH --partition main
#SBATCH --time 6-23:59:00
#SBATCH --job-name {job}
#SBATCH --output {out_dir}/%x.out
#SBATCH --mail-user={mail}
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus={gpu}
#SBATCH --mem={mem}

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
nvidia-smi -L

module load anaconda
source activate {env}

cd {repo}
python -u tuning/run_tuning.py --model {model} --dataset {dataset}
"""


def build():
    written = 0
    for model in MODELS:
        for dataset in DATASETS:
            if dataset == "MNLI" and model in OFFICIAL_MNLI:
                continue                      # official checkpoint -> no tuning job
            big = model in OFFICIAL_MNLI      # same set = the three large models
            (SBATCH_DIR / f"tune_{model}__{dataset}.sbatch").write_text(
                TEMPLATE.format(
                    model=model, dataset=dataset,
                    fname=f"tune_{model}__{dataset}.sbatch",
                    job=f"tune_{model}__{dataset}",
                    gpu=GPU_LARGE if big else GPU_BASE,
                    mem="32G" if big else "16G",
                    out_dir=OUT_DIR, mail=MAIL, env=ENV, repo=REPO))
            written += 1
    print(f"{written} sbatch files written to {SBATCH_DIR} "
          f"(the 3 official-checkpoint MNLI combinations are skipped)")


if __name__ == "__main__":
    build()
