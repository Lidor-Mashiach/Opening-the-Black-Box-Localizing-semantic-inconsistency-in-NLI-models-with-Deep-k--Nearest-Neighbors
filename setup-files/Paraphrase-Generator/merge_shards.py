"""Merge sharded paraphrase parts into the final bank (+ reset derived data).

Parallel SLURM array tasks each write paraphrase_bank.partIIIofNNN.csv for a
disjoint slice of a dataset's hypotheses (see generate_paraphrases.py
--num-shards / --shard-index). This assembles them into the single, canonical

    Datasets/<dataset>/paraphrases/paraphrase_bank.csv

concatenated, defensively de-duplicated, and SORTED by (original hypothesis
index, para_idx) so the file is deterministic and reproducible. It then merges
the per-shard stats into one bank-level paraphrase_bank_stats.json and -
because a new complete bank invalidates everything downstream - performs the
SAME derived-data reset the non-sharded generator does, exactly once.

Ordering / leakage note: the pipeline's val/test split is HASH-based
(zlib.crc32(pair_id) in common/alignment.py), so row ORDER never affects which
side a hypothesis lands on. Sorting here is purely for a clean, reproducible
file, not a correctness requirement, and disjoint shards can never share a
pair_id, so concatenation can never cause leakage or duplication.

Run:  python merge_shards.py --datasets SNLI       (after all SNLI shards finish)
      python merge_shards.py                        (every dataset that has parts)
      python merge_shards.py --datasets SNLI --rebuild
      python merge_shards.py --datasets SNLI --keep-parts
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import pandas as pd

from common.logging_utils import banner, log
import generate_paraphrases as gp   # same directory - reuse registry, safe-writer, reset

PART_RE = re.compile(r"paraphrase_bank\.part(\d+)of(\d+)\.csv$")


def _merge_stats(part_paths, final_csv, bank_df):
    """Aggregate the per-shard stats.json files into one bank-level stats."""
    sums = {"pool_size": 0, "hypotheses_kept": 0, "hypotheses_at_full_quota": 0,
            "hypotheses_partial": 0, "hypotheses_dropped_zero_verified": 0}
    gate, count_hist, fill_hist, meta = {}, {}, {}, {}
    for p in part_paths:
        sp = p.with_name(p.name[:-4] + "_stats.json")
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        meta = meta or {k: s.get(k) for k in (
            "dataset", "split", "generator_model", "verifier_model",
            "target_paraphrases_per_hypothesis", "policy",
            "length_ratio_bounds", "max_generation_rounds")}
        for k in sums:
            sums[k] += int(s.get(k, 0) or 0)
        for k, v in (s.get("gate_rejections") or {}).items():
            gate[k] = gate.get(k, 0) + int(v)
        for k, v in (s.get("paraphrase_count_histogram") or {}).items():
            count_hist[k] = count_hist.get(k, 0) + int(v)
        for k, v in (s.get("fill_rounds_histogram") or {}).items():
            fill_hist[k] = fill_hist.get(k, 0) + int(v)

    length_stats = None
    if len(bank_df):
        hw = bank_df["hypothesis"].astype(str).str.split().str.len()
        pw = bank_df["paraphrase"].astype(str).str.split().str.len()
        length_stats = {
            "hypothesis_mean_words": round(float(hw.mean()), 2),
            "hypothesis_std_words": round(float(hw.std()), 2),
            "paraphrase_mean_words": round(float(pw.mean()), 2),
            "paraphrase_std_words": round(float(pw.std()), 2),
            "mean_length_ratio": round(float((pw / hw).mean()), 3),
        }

    stats = {**meta,
             "merged_from_shards": len(part_paths),
             "pool_size": sums["pool_size"],
             "hypotheses_kept": sums["hypotheses_kept"],
             "hypotheses_at_full_quota": sums["hypotheses_at_full_quota"],
             "hypotheses_partial": sums["hypotheses_partial"],
             "paraphrase_count_histogram": {k: count_hist[k]
                                            for k in sorted(count_hist, key=int)},
             "fill_rounds_histogram": {k: fill_hist[k]
                                       for k in sorted(fill_hist, key=int)},
             "hypotheses_dropped_zero_verified": sums["hypotheses_dropped_zero_verified"],
             "paraphrase_rows": len(bank_df),
             "gate_rejections": gate,
             "length_stats": length_stats,
             "output_csv": str(final_csv)}
    (final_csv.parent / "paraphrase_bank_stats.json").write_text(json.dumps(stats, indent=2))


def merge_dataset(dataset_key, rebuild=False, keep_parts=False):
    final_csv = gp.registry_bank_csv(dataset_key)
    if final_csv.exists() and not rebuild:
        log("SKIP", f"final bank already exists at {final_csv} - use --rebuild "
            f"to reassemble from shards", dataset=dataset_key)
        return

    parts_dir = final_csv.parent
    part_paths = sorted(p for p in parts_dir.glob("paraphrase_bank.part*of*.csv")
                        if PART_RE.search(p.name))
    if not part_paths:
        log("WARN", f"no shard parts (paraphrase_bank.partIIIofNNN.csv) in "
            f"{parts_dir} - nothing to merge", dataset=dataset_key)
        return

    num_shards = int(PART_RE.search(part_paths[0].name).group(2))
    have = sorted(int(PART_RE.search(p.name).group(1)) for p in part_paths)
    missing = [i for i in range(num_shards) if i not in have]
    if missing:
        log("WARN", f"expected {num_shards} shards but {len(missing)} missing "
            f"{missing} - merging only the {len(part_paths)} present (RE-RUN the "
            f"missing shards, then merge again, for the complete bank)",
            dataset=dataset_key)

    banner("BANK", f"merging {len(part_paths)} shard part(s)", dataset=dataset_key)
    merged = pd.concat([pd.read_csv(p) for p in part_paths], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["pair_id", "para_idx"])
    # SORT for a deterministic, reproducible file. Order does NOT affect the
    # hash-based val/test split (zlib.crc32(pair_id) in common/alignment.py),
    # so this is cosmetic + reproducible - never a correctness/leakage
    # requirement (disjoint shards cannot share a pair_id).
    if len(merged):
        order = merged["pair_id"].astype(str).str.extract(r"-(\d+)$")[0].astype(int)
        merged = (merged.assign(_o=order).sort_values(["_o", "para_idx"])
                  .drop(columns="_o").reset_index(drop=True))

    gp._safe_write_bank(merged, final_csv)
    log("BANK", f"assembled bank -> {final_csv} ({len(merged)} rows"
        + (f", {before - len(merged)} duplicate rows dropped"
           if before != len(merged) else "") + ")", dataset=dataset_key)
    _merge_stats(part_paths, final_csv, merged)

    # a complete new bank invalidates everything downstream -> reset ONCE
    banner("CLEAN", f"bank assembled for {dataset_key} - resetting the derived "
           f"pipeline (Runtime-Data + Step results)")
    gp.reset_derived_data(dataset_key=dataset_key)

    if not keep_parts and not missing:
        for p in part_paths:
            p.unlink(missing_ok=True)
            p.with_name(p.name[:-4] + "_stats.json").unlink(missing_ok=True)
        log("CLEAN", f"removed {len(part_paths)} shard part file(s) "
            f"(pass --keep-parts to retain them)", dataset=dataset_key)
    elif missing:
        log("BANK", "shard parts kept (some shards were missing) - re-run them "
            "and merge again", dataset=dataset_key)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of the DATASETS registry "
                             "(default: every dataset that has shard parts)")
    parser.add_argument("--rebuild", action="store_true",
                        help="reassemble even if the final bank exists (the "
                             "previous bank is kept as paraphrase_bank.backup.csv)")
    parser.add_argument("--keep-parts", action="store_true",
                        help="do NOT delete the shard part files after merging")
    args = parser.parse_args()
    datasets = args.datasets.split(",") if args.datasets else list(gp.DATASETS)
    for dataset_key in datasets:
        if dataset_key not in gp.DATASETS:
            raise SystemExit(f"[merge] unknown dataset '{dataset_key}'; known: "
                             f"{', '.join(gp.DATASETS)}")
        merge_dataset(dataset_key, rebuild=args.rebuild, keep_parts=args.keep_parts)


if __name__ == "__main__":
    main()
