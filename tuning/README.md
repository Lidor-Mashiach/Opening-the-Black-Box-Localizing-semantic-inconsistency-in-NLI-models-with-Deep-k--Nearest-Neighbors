# 🎛️ Tuning - Optuna Hyper-Parameter Search

One Optuna study per (model, dataset) combination - 12 studies total - each
runnable as a single week-long GPU job on the BGU SLURM cluster, fully
resumable after crashes or pre-emption.

## 🎯 Objective (per combination)

**Maximize the validation accuracy of the fine-tuned NLI classifier** - the
same quantity Step-1 reports, measured with the same shared `predict()`.
Each trial fine-tunes on a seeded train subsample and scores a seeded
validation subsample (`tuning.train_subsample` / `tuning.val_subsample` in
`configs/base.yaml`), so a full 20-trial study fits one job.

## 🔍 Search Space (one entry per feature)

Defined declaratively in `configs/base.yaml -> tuning.search_space`:

| Feature | Type / Range | Rationale |
|---------|--------------|-----------|
| `learning_rate` | float, **log**-uniform `5e-6 .. 5e-5` | Brackets the standard fine-tuning LRs; log scale because LR effects are multiplicative |
| `weight_decay` | float, uniform `0 .. 0.1` | From no regularization up to the common AdamW ceiling |
| `warmup_ratio` | float, uniform `0 .. 0.2` | No warmup up to 20% of steps |
| `epochs` | int, `2 .. 3` | NLI fine-tuning saturates fast; >3 mostly overfits |
| `batch_size` | categorical: `[16, 32]` for base models, `[8, 16]` for the large ones | Bounded by GPU VRAM; options centered on each model's default |

Sampler: TPE (seeded). Direction: maximize.

## 📁 Layout

```
tuning/
├── optuna_search.py      # objective + study logic (see its docstring)
├── run_tuning.py         # CLI: one study (sbatch mode) or all 12 sequentially
├── run_all_sbatch.sh     # MAIN RUNNER: submits every uncommented sbatch job
├── sbatch/               # 12 job files: tune_<MODEL>__<DATASET>.sbatch
├── outputs/              # SLURM .out logs land here (%x-%J.out)
└── results/<COMBO>/      # optuna.db (resumable), best_params.yaml, trials.csv
```

## 🔄 Where the Results Go

The best trial is written to **`configs/tuned/<MODEL>__<DATASET>.yaml`** -
`load_config()` merges it automatically on top of base -> model -> dataset.
To retrain with the tuned values:

```bash
python Research-Pipeline/run_pipeline.py --force
```

(`--force` cascades: retrain -> refilter -> re-encode -> regenerate
paraphrases -> re-infer -> re-analyze.)

## 🚀 Running on the BGU Cluster

```bash
# one-time, inside the repo on the cluster:
conda create -n nli_dknn python=3.10
conda activate nli_dknn
pip install -r requirements.txt

# edit the two CHANGE-ME lines in tuning/sbatch/*.sbatch
#   (conda env name + repo path), then:
cd tuning
bash run_all_sbatch.sh        # comment out unwanted jobs in the JOBS list first
squeue --me                   # monitor;  scancel <id> to cancel
```

Every job: `--partition main`, `--time 6-23:59:00` (just under the 7-day
cap), `--gpus=1`, `--mem=32G`, live logs via `PYTHONUNBUFFERED`. GPU/CPU is
auto-detected in code (`torch.cuda.is_available()`) - the same scripts run
locally without SLURM:

```bash
python tuning/run_tuning.py --model BERT-base --dataset SNLI
```

## 🛟 Crash Safety

| Layer | Mechanism |
|-------|-----------|
| Optuna study | sqlite storage + `load_if_exists` - resubmit and it continues from the next trial |
| Step-1 full training | rolling epoch checkpoint + `resume_from_checkpoint` (see Step-1 README) |
| Every pipeline stage | skips itself when its output already exists |
