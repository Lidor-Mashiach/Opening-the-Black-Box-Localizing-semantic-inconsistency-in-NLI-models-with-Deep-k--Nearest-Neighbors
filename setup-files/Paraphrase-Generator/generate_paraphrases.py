"""Build the DATASET-level paraphrase bank - a static, shareable asset.

One bank per dataset (model-independent), created ONCE from the raw train
split - exactly the design a research community can reuse:

1. POOL     deterministic_pool_sample() draws paraphrases.pool_size
            hypotheses from the raw train split (seeded; pair_ids identical
            to the ones the whole pipeline uses).
2. GENERATE the leading open paraphraser (humarin/chatgpt_paraphraser_on_T5_base)
            produces candidates_per_hypothesis rewordings per hypothesis
            (nucleus / top-p sampling, run in bf16).
3. LENGTH   a candidate must stay at the hypothesis's own level: its
            word-count ratio vs the hypothesis must be inside
            paraphrases.length_ratio (default 0.6-1.5). Each dataset's
            difficulty therefore stays measurable - short SNLI captions get
            short paraphrases, long ANLI hypotheses get long ones.
4. VERIFY   a DOUBLE gate with the strongest public NLI model
            (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli):
              (a) equivalence - hypothesis -> candidate AND candidate ->
                  hypothesis are BOTH 'entailment' (symmetric equivalence);
              (b) relation    - the verifier's label for
                  (premise, candidate) must EQUAL the gold label.
            Together: same meaning as the hypothesis AND the same logical
            relation to the premise - Premise:Hypothesis == Premise:Paraphrase.
5. RETRY    hypotheses still short of the target get MORE generation rounds -
            FRESH nucleus sampling every round (new seed + a mild temperature
            ramp, deduplicated against everything already tried) - bounded by
            max_generation_rounds (default 6), with an early stop after
            early_stop_patience (2) consecutive rounds that add nothing.
6. QUOTA    UP TO per_hypothesis (default 3) verified paraphrases per
            hypothesis, kept partial: a hypothesis with 1..per_hypothesis-1
            verified paraphrases is kept as-is. Only a hypothesis with ZERO
            verified paraphrases is dropped (loudly reported in the stats);
            raise max_generation_rounds / candidates_per_hypothesis to reduce
            partials and zero-verified drops.

PROTECTION - the bank is treated like the raw data: running this script (or
the pipeline) NEVER overwrites an existing bank. Regeneration requires an
explicit --rebuild, and even then the previous bank is preserved as
paraphrase_bank.backup.csv and the new one is written atomically (tmp file ->
replace), so a crash mid-generation can never destroy the existing bank.

Output:  Datasets/<dataset>/paraphrases/paraphrase_bank.csv
         (pair_id, premise, hypothesis, paraphrase, label, para_idx)
         + paraphrase_bank_stats.json

Per-model usage happens later, in Step-2: each model consumes only the rows
whose hypothesis IT classified correctly - so all models share the exact same
paraphrases for shared hypotheses, and comparisons are apples-to-apples.

Run:  python generate_paraphrases.py                     (banks that are missing)
      python generate_paraphrases.py --datasets SNLI
      python generate_paraphrases.py --rebuild            (explicit regeneration)
      python generate_paraphrases.py --limit 50           (quick trial run)

Parallel (SLURM array) - split one dataset across N shards, then assemble:
      python generate_paraphrases.py --datasets SNLI --num-shards 3 --shard-index 0
      python generate_paraphrases.py --datasets SNLI --num-shards 3 --shard-index 1
      python generate_paraphrases.py --datasets SNLI --num-shards 3 --shard-index 2
      python merge_shards.py --datasets SNLI              (after all shards finish)
Each shard writes paraphrase_bank.partIIIofNNN.csv (disjoint, never colliding);
merge_shards.py concatenates + sorts them into the final bank and resets the
derived pipeline once. Shard runs do NOT reset derived data themselves.

GPU strongly recommended. The bank depends only on the raw data + the two
public models above - retraining YOUR models never invalidates it.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # setup-files/Paraphrase-Generator/ -> repo root
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from common import data_loading
from common.config_loader import dataset_dir, load_config
from common.gpu import resolve_device
from common.logging_utils import banner, log
from common.sampling import deterministic_pool_sample

PARAPHRASE_CONFIG = Path(__file__).resolve().parent / "paraphrase_config.yaml"

# =====================================================================
# GLOBAL KNOBS - the ONLY two things to touch when scaling the banks
# =====================================================================
# Target verified paraphrases per hypothesis (ASPIRE to this). Partial is kept:
# a hypothesis with 1..N-1 verified paraphrases stays as-is; only a hypothesis
# with ZERO verified paraphrases is dropped from the bank.
PARAPHRASES_PER_HYPOTHESIS = 5

# Dataset registry. KEY = dataset name; VALUE = where the dataset lives,
# relative to the repo root. The bank is written to
#     <VALUE>/paraphrases/paraphrase_bank.csv
# To ADD a dataset: add one line here + a small configs/datasets/<KEY>.yaml
# (hf_id, dir, splits - the pipeline needs it too). Nothing else.
DATASETS = {
    "SNLI": "Datasets/SNLI-Stanford_NLI",
    "MNLI": "Datasets/MNLI-MultiGenre_NLI",
    "ANLI": "Datasets/ANLI-Adversarial_NLI",
}


def registry_bank_csv(dataset_key):
    """The bank path, driven by the global DATASETS registry."""
    return REPO_ROOT / DATASETS[dataset_key] / "paraphrases" / "paraphrase_bank.csv"


def load_generator_config(dataset_key):
    """Dataset config (hf_id, splits, seed) + this module's own
    paraphrase_config.yaml. The two GLOBALS above drive the quota and the
    output location; a mismatch with configs/ aborts loudly, because the
    pipeline reads the bank from the configs/ location."""
    if dataset_key not in DATASETS:
        raise SystemExit(
            f"[paraphrases] unknown dataset '{dataset_key}'. The global "
            f"DATASETS registry knows: {', '.join(DATASETS)}. Add it there "
            f"(plus a small configs/datasets/{dataset_key}.yaml).")
    cfg = load_config(dataset_key=dataset_key)
    with open(PARAPHRASE_CONFIG, "r", encoding="utf-8") as f:
        cfg["paraphrases"] = yaml.safe_load(f)["paraphrases"]
    cfg["paraphrases"]["per_hypothesis"] = PARAPHRASES_PER_HYPOTHESIS

    registry_dir = (REPO_ROOT / DATASETS[dataset_key]).resolve()
    if registry_dir != dataset_dir(cfg).resolve():
        log("ERROR", f"the DATASETS registry points to {registry_dir} but "
            f"configs/datasets/{dataset_key}.yaml points to "
            f"{dataset_dir(cfg)} - the PIPELINE reads the latter. Update the "
            f"yaml's `dir` to match the registry.", dataset=dataset_key)
        raise SystemExit(1)
    return cfg


def _normalize(text):
    """Comparison key for duplicate detection (case / spacing / final period)."""
    return " ".join(str(text).lower().split()).rstrip(" .?!")


def _length_ok(hypothesis, candidate, ratio_min, ratio_max):
    """Keep length / complexity at the hypothesis's own level.

    The candidate's word count divided by the hypothesis's word count must
    stay inside [ratio_min, ratio_max] - so paraphrase difficulty tracks the
    dataset's difficulty and stays measurable.
    """
    h_words = max(len(str(hypothesis).split()), 1)
    c_words = len(str(candidate).split())
    ratio = c_words / h_words
    return ratio_min <= ratio <= ratio_max


CANONICAL_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


def canonical_label_map(id2label):
    """{verifier index -> canonical dataset label} by name; None if unmappable."""
    mapping = {}
    for idx, name in (id2label or {}).items():
        for canonical, target in CANONICAL_LABELS.items():
            if canonical in str(name).lower():
                mapping[int(idx)] = target
                break
    return mapping if len(mapping) == 3 else None


def _safe_write_bank(df, out_csv):
    """Crash-safe, non-destructive write.

    The new bank goes to a temp file first; only after it is fully written is
    the previous bank (if any) moved to paraphrase_bank.backup.csv and the
    temp file atomically promoted. The existing bank is therefore intact at
    every moment until the new one is complete.
    """
    tmp = out_csv.with_name("paraphrase_bank.tmp.csv")
    df.to_csv(tmp, index=False)
    if out_csv.exists():
        backup = out_csv.with_name("paraphrase_bank.backup.csv")
        out_csv.replace(backup)
        log("BANK", f"previous bank preserved -> {backup}")
    tmp.replace(out_csv)


class ParaphraseGenerator:
    """Paraphrase candidates from a seq2seq model - two decoding modes."""

    def __init__(self, cfg, device):
        model_id = cfg["paraphrases"]["generator_model"]
        print(f"[generator] loading {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        gen_dtype = (torch.bfloat16 if device.type == "cuda"
                     and torch.cuda.is_bf16_supported() else torch.float32)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=gen_dtype).to(device).eval()
        self.prefix = cfg["paraphrases"]["generator_prompt_prefix"]
        self.n_candidates = cfg["paraphrases"]["candidates_per_hypothesis"]
        self.max_len_ratio = float(cfg["paraphrases"]["length_ratio"]["max"])
        self.device = device

    @torch.no_grad()
    def generate(self, hypotheses, temperature=1.0):
        """list[str] -> list[list[str]] of n_candidates rewordings each.

        Nucleus sampling (top-p) for EVERY round. Diverse beam search with
        num_beams=n_candidates was the source of the CUDA-OOM thrash and most
        of the runtime (it expands the batch by n_candidates AND decodes each
        beam group sequentially). Sampling gives all n_candidates in one packed
        forward, uses a fraction of the memory, and gets its diversity from
        top-p + the per-round temperature ramp - and since the double verifier
        gates every kept paraphrase, quality is unchanged; only wasted compute
        is removed.
        """
        inputs = [self.prefix + h for h in hypotheses]
        enc = self.tokenizer(inputs, padding=True, truncation=True,
                             max_length=128, return_tensors="pt").to(self.device)
        # A paraphrase stays at the hypothesis's own length: the length gate
        # rejects anything above length_ratio.max WORDS, so bound NEW tokens to
        # that same ratio over the input (max input in the batch, so the longest
        # hypothesis is never truncated) + a small margin. Tokens >= words, so
        # this can never cut a gate-passing paraphrase; it just avoids the wasted
        # decode steps of a fixed 128. Tie to the config so the two stay in sync.
        max_new = int(enc["input_ids"].shape[1] * self.max_len_ratio) + 8
        out = self.model.generate(
            **enc, do_sample=True, top_p=0.95, temperature=temperature,
            num_return_sequences=self.n_candidates,
            no_repeat_ngram_size=2, max_new_tokens=max_new)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        n = self.n_candidates
        return [decoded[i * n:(i + 1) * n] for i in range(len(hypotheses))]


class EntailmentVerifier:
    """The double gate: symmetric equivalence + premise-relation preservation."""

    def __init__(self, cfg, device):
        model_id = cfg["paraphrases"]["verifier_model"]
        print(f"[verifier] loading {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        ver_dtype = (torch.bfloat16 if device.type == "cuda"
                     and torch.cuda.is_bf16_supported() else torch.float32)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=ver_dtype).to(device).eval()
        self.batch_size = cfg["paraphrases"]["verifier_batch_size"]
        self.device = device
        id2label = getattr(self.model.config, "id2label", None) or {}
        label_map = canonical_label_map(id2label)
        if label_map is None:
            raise ValueError("verifier id2label does not name entailment/neutral/"
                             "contradiction - choose a verifier with proper label names")
        self._lookup = np.arange(len(label_map))
        for src, dst in label_map.items():
            self._lookup[src] = dst
        print(f"[verifier] label map (verifier -> canonical): {label_map}")

    @torch.no_grad()
    def _predict_canonical(self, first, second):
        """3-way predictions for (first[i], second[i]), in the CANONICAL
        convention (0=entailment, 1=neutral, 2=contradiction)."""
        preds = []
        for start in range(0, len(first), self.batch_size):
            end = min(start + self.batch_size, len(first))
            enc = self.tokenizer(list(first[start:end]), list(second[start:end]),
                                 padding=True, truncation=True, max_length=128,
                                 return_tensors="pt").to(self.device)
            batch = self.model(**enc).logits.argmax(dim=-1).cpu().numpy()
            preds.append(self._lookup[batch])
        return np.concatenate(preds)

    def verify_batch(self, premises, hypotheses, candidates, gold_labels):
        """Both gates in ONE fused forward pass over 3x the pairs.

        Instead of three separate verifier calls (forward equivalence,
        backward equivalence, premise-relation), concatenate all three pair
        sets and score them in a single batched sweep - the GPU sees 3x the
        work per launch, which is far more efficient than three smaller
        sweeps. Returns a boolean mask: candidate passes BOTH gates.
        """
        if not candidates:
            return np.array([], dtype=bool)
        n = len(candidates)
        # one big list: [fwd pairs ... | bwd pairs ... | relation pairs ...]
        first = list(hypotheses) + list(candidates) + list(premises)
        second = list(candidates) + list(hypotheses) + list(candidates)
        preds = self._predict_canonical(first, second)
        forward = preds[:n] == CANONICAL_LABELS["entailment"]
        backward = preds[n:2 * n] == CANONICAL_LABELS["entailment"]
        relation = preds[2 * n:] == np.asarray(gold_labels)
        equivalent = forward & backward
        return equivalent, relation

    def both_entail(self, hypotheses, candidates):
        """Gate (a): hypothesis <-> candidate entail EACH OTHER.

        Kept for the sanity check / direct callers; the hot path uses
        verify_batch, which fuses this with gate (b)."""
        if not candidates:
            return np.array([], dtype=bool)
        forward = self._predict_canonical(hypotheses, candidates) == CANONICAL_LABELS["entailment"]
        backward = self._predict_canonical(candidates, hypotheses) == CANONICAL_LABELS["entailment"]
        return forward & backward

    def relation_matches(self, premises, candidates, gold_labels):
        """Gate (b): the (premise, candidate) relation equals the gold label."""
        if not candidates:
            return np.array([], dtype=bool)
        predicted = self._predict_canonical(premises, candidates)
        return predicted == np.asarray(gold_labels)


def run_verification_round(records, generator, verifier, verified, tried,
                           per_hyp, gen_bs, use_sampling, length_bounds,
                           counters, temperature=1.2):
    """One generate -> length-gate -> double-verify pass; updates `verified`.

    Gates, in order (rejection reasons accumulate in `counters`):
      length      word-count ratio outside paraphrases.length_ratio
      equivalence hypothesis <-> candidate not mutually entailing
      relation    (premise, candidate) label differs from the gold label
    """
    ratio_min, ratio_max = length_bounds
    for start in range(0, len(records), gen_bs):
        batch = records[start:start + gen_bs]
        all_candidates = generator.generate([r["hypothesis"] for r in batch],
                                            temperature=temperature)
        cleaned = []
        for record, cands in zip(batch, all_candidates):
            pid = record["pair_id"]
            kept = []
            for cand in cands:
                cand = cand.strip()
                key = _normalize(cand)
                if not cand or key in tried[pid]:
                    continue
                tried[pid].add(key)
                if not _length_ok(record["hypothesis"], cand, ratio_min, ratio_max):
                    counters["rejected_length"] += 1
                    continue
                kept.append(cand)
            cleaned.append(kept)

        flat_h = [r["hypothesis"] for r, cs in zip(batch, cleaned) for _ in cs]
        flat_p = [r["premise"] for r, cs in zip(batch, cleaned) for _ in cs]
        flat_g = [r["label"] for r, cs in zip(batch, cleaned) for _ in cs]
        flat_c = [c for cs in cleaned for c in cs]
        equivalent, relation_ok = verifier.verify_batch(flat_p, flat_h, flat_c, flat_g)
        counters["rejected_equivalence"] += int((~equivalent).sum())
        counters["rejected_relation"] += int((equivalent & ~relation_ok).sum())
        ok = equivalent & relation_ok

        pos = 0
        for record, cands in zip(batch, cleaned):
            pid = record["pair_id"]
            for j, cand in enumerate(cands):
                if ok[pos + j] and len(verified[pid]) < per_hyp:
                    verified[pid].append(cand)
                    counters["accepted"] += 1
            pos += len(cands)
        done = min(start + gen_bs, len(records))
        mode = "retry" if use_sampling else "round-1"
        print(f"  [{mode}] {done}/{len(records)} hypotheses", flush=True)


def fill_quota_over_rounds(records, generator, verifier, verified, tried,
                           per_hyp, gen_bs, length_bounds, counters,
                           max_rounds, seed, dataset_key, patience=2):
    """Fill each hypothesis toward the quota, then STOP intelligently.

    Round 1 uses diverse beam search (deterministic, highest quality); every
    following round draws FRESH candidates via nucleus sampling - a new seed
    and a mild temperature ramp per round, deduplicated against everything
    already tried for that hypothesis - and runs the exact same gates. A
    hypothesis leaves the loop the moment it holds per_hyp verified
    paraphrases.

    PERFORMANCE GUARD (early stop): the hard remainder of a large split can
    resist paraphrasing no matter how many rounds run. If `patience`
    consecutive retry rounds add ZERO new verified paraphrases across all
    still-short hypotheses, the generator is stuck and the loop stops instead
    of grinding to max_rounds. Whatever was collected is kept (partial rows
    are allowed by the caller). Returns {pair_id: round at which quota met}.
    """
    fill_round = {}

    def _mark(round_idx):
        for record in records:
            pid = record["pair_id"]
            if pid not in fill_round and len(verified[pid]) >= per_hyp:
                fill_round[pid] = round_idx

    run_verification_round(records, generator, verifier, verified, tried,
                           per_hyp, gen_bs, use_sampling=False,
                           length_bounds=length_bounds, counters=counters,
                           temperature=1.0)
    _mark(1)
    short = [r for r in records if len(verified[r["pair_id"]]) < per_hyp]
    round_idx = 1
    stagnant = 0
    while short and round_idx < max_rounds:
        round_idx += 1
        accepted_before = counters["accepted"]
        torch.manual_seed(seed + round_idx)      # fresh candidates every round
        temperature = min(1.2 + 0.05 * (round_idx - 2), 1.5)
        log("BANK", f"round {round_idx}/{max_rounds}: {len(short)} hypotheses "
            f"still below quota - fresh sampling (temperature {temperature:.2f})",
            dataset=dataset_key)
        run_verification_round(short, generator, verifier, verified, tried,
                               per_hyp, gen_bs, use_sampling=True,
                               length_bounds=length_bounds, counters=counters,
                               temperature=temperature)
        _mark(round_idx)
        gained = counters["accepted"] - accepted_before
        stagnant = stagnant + 1 if gained == 0 else 0
        if stagnant >= patience:
            log("WARN", f"early stop: {patience} consecutive rounds added no "
                f"new paraphrases - {len(short)} hypothesis(es) stay below "
                f"quota (their partial paraphrases are kept)",
                dataset=dataset_key)
            break
        short = [r for r in records if len(verified[r["pair_id"]]) < per_hyp]
    return fill_round


def _shard_paths(dataset_key, shard_index, num_shards):
    """Output CSV + stats path for a shard (or the final bank when not sharded)."""
    final = registry_bank_csv(dataset_key)
    if num_shards and num_shards > 1:
        csv = final.with_name(
            f"paraphrase_bank.part{shard_index:03d}of{num_shards:03d}.csv")
        stats = csv.with_name(csv.name[:-4] + "_stats.json")
    else:
        csv = final
        stats = csv.parent / "paraphrase_bank_stats.json"
    return csv, stats


def generate_for_dataset(dataset_key, generator, verifier, cfg, limit=None,
                         rebuild=False, shard_index=None, num_shards=None):
    sharded = bool(num_shards and num_shards > 1)
    out_csv, stats_path = _shard_paths(dataset_key, shard_index, num_shards)
    if out_csv.exists() and not rebuild:
        what = f"shard {shard_index + 1}/{num_shards}" if sharded else "bank"
        log("SKIP", f"{what} already exists at {out_csv} - protected like the "
            f"raw data; regenerate only with --rebuild", dataset=dataset_key)
        return None

    banner("BANK", "building the static paraphrase bank"
           + (f" (shard {shard_index + 1}/{num_shards})" if sharded else ""),
           dataset=dataset_key)
    train = data_loading.load_nli(cfg, "train")
    train = data_loading.add_pair_ids(train, cfg, "train")
    df = train.to_pandas()[["pair_id", "premise", "hypothesis", "label"]]
    pool = deterministic_pool_sample(df, cfg)
    if limit is not None:
        pool = pool.head(limit)
    if sharded:                     # disjoint contiguous slice of the pool for THIS shard
        n_pool = len(pool)
        start = shard_index * n_pool // num_shards
        end = (shard_index + 1) * n_pool // num_shards
        pool = pool.iloc[start:end]
        log("BANK", f"shard {shard_index + 1}/{num_shards}: hypotheses "
            f"[{start}:{end}) of {n_pool}", dataset=dataset_key)
    per_hyp = cfg["paraphrases"]["per_hypothesis"]
    gen_bs = cfg["paraphrases"]["generation_batch_size"]
    log("BANK", f"pool: {len(pool)} hypotheses, target: up to {per_hyp} "
        f"paraphrases each (partial kept)", dataset=dataset_key)

    length_bounds = (float(cfg["paraphrases"]["length_ratio"]["min"]),
                     float(cfg["paraphrases"]["length_ratio"]["max"]))
    counters = {"rejected_length": 0, "rejected_equivalence": 0,
                "rejected_relation": 0, "accepted": 0}
    records = pool.to_dict("records")
    verified = {r["pair_id"]: [] for r in records}
    tried = {r["pair_id"]: {_normalize(r["hypothesis"])} for r in records}

    max_rounds = int(cfg["paraphrases"].get("max_generation_rounds", 8))
    patience = int(cfg["paraphrases"].get("early_stop_patience", 2))
    fill_round = fill_quota_over_rounds(records, generator, verifier, verified,
                                        tried, per_hyp, gen_bs, length_bounds,
                                        counters, max_rounds, cfg["seed"],
                                        dataset_key, patience=patience)

    rows, dropped, partial = [], 0, 0
    count_hist = {}
    for record in records:
        pid = record["pair_id"]
        keep = verified[pid]
        n = len(keep)
        count_hist[n] = count_hist.get(n, 0) + 1
        if n == 0:                       # nothing verified at all -> cannot use
            dropped += 1
            continue
        if n < per_hyp:                  # keep PARTIAL (1..per_hyp-1) rather than lose it
            partial += 1
        for para_idx, paraphrase in enumerate(keep[:per_hyp]):
            rows.append({"pair_id": pid,
                         "premise": record["premise"],
                         "hypothesis": record["hypothesis"],
                         "paraphrase": paraphrase,
                         "label": record["label"],
                         "para_idx": para_idx})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bank_df = pd.DataFrame(rows)
    if sharded:
        bank_df.to_csv(out_csv, index=False)   # part file; merge_shards.py assembles the bank safely
    else:
        _safe_write_bank(bank_df, out_csv)

    # difficulty stays measurable: paraphrase length must track the hypothesis
    if len(bank_df):
        hyp_words = bank_df["hypothesis"].astype(str).str.split().str.len()
        para_words = bank_df["paraphrase"].astype(str).str.split().str.len()
        length_stats = {
            "hypothesis_mean_words": round(float(hyp_words.mean()), 2),
            "hypothesis_std_words": round(float(hyp_words.std()), 2),
            "paraphrase_mean_words": round(float(para_words.mean()), 2),
            "paraphrase_std_words": round(float(para_words.std()), 2),
            "mean_length_ratio": round(float((para_words / hyp_words).mean()), 3),
        }
    else:
        length_stats = None

    kept = len(records) - dropped
    stats = {
        "dataset": dataset_key,
        "split": "train",
        "generator_model": cfg["paraphrases"]["generator_model"],
        "verifier_model": cfg["paraphrases"]["verifier_model"],
        "target_paraphrases_per_hypothesis": per_hyp,
        "policy": "keep partial (>=1); only 0-verified hypotheses are dropped",
        "length_ratio_bounds": list(length_bounds),
        "pool_size": len(records),
        "max_generation_rounds": max_rounds,
        "hypotheses_kept": kept,
        "hypotheses_at_full_quota": count_hist.get(per_hyp, 0),
        "hypotheses_partial": partial,
        "paraphrase_count_histogram": {str(k): count_hist[k]
                                       for k in sorted(count_hist)},
        "fill_rounds_histogram": {str(r): sum(1 for v in fill_round.values() if v == r)
                                  for r in sorted(set(fill_round.values()))},
        "hypotheses_dropped_zero_verified": dropped,
        "paraphrase_rows": len(rows),
        "gate_rejections": {k: v for k, v in counters.items() if k != "accepted"},
        "length_stats": length_stats,
        "output_csv": str(out_csv),
        "shard": ({"index": shard_index, "num_shards": num_shards}
                  if sharded else None),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    if partial or dropped:
        log("WARN", f"{count_hist.get(per_hyp, 0)} hypotheses reached the full "
            f"{per_hyp}; {partial} kept with fewer (partial); {dropped} dropped "
            f"(zero verified). Partial rows are intentionally KEPT.",
            dataset=dataset_key)
    log("BANK", f"bank saved: {kept} hypotheses, {len(rows)} paraphrase rows "
        f"(full quota: {count_hist.get(per_hyp, 0)}, partial: {partial}, "
        f"dropped: {dropped}) -> {out_csv}",
        dataset=dataset_key)
    return stats


# Everything a previous run/experiment DERIVED. New paraphrase banks mean a
# fresh study from zero, so these are cleared before generation. The tuned
# Optuna configs (configs/tuned/) and the downloaded backbones (Models/) are
# NOT here - they are independent of the paraphrase banks and must survive.
DERIVED_LOCATIONS = [
    ("Runtime-Data (reduced copies + per-layer encodings)", "Runtime-Data"),
    ("Step-1 results", "Research-Pipeline/PhaseA/Step-1_Train-Filter-Encode/results"),
    ("Step-2 results", "Research-Pipeline/PhaseA/Step-2_Paraphrase-Inference/results"),
    ("Step-3 results", "Research-Pipeline/PhaseA/Step-3_DkNN-and-Layer-Distance/results"),
    ("Step-4 results", "Research-Pipeline/PhaseA/Step-4_Consistency-and-Diagnosis/results"),
]


def reset_derived_data(dataset_key=None):
    """Clear stale derived artifacts so a fresh bank starts from zero.

    dataset_key is None  -> full reset (the main-setup run builds ALL datasets
        in sequence, so wiping everything derived is correct).
    dataset_key is set   -> reset ONLY that dataset's derived data, by matching
        the '<MODEL>__<DATASET>' / '<DATASET>' path segments. This is what makes
        parallel per-dataset sbatch jobs safe: an SNLI job never deletes what a
        concurrent MNLI/ANLI job is building.

    Always preserves each folder's tracked README; never touches configs/tuned
    or Models.
    """
    import shutil
    found = False

    def _dataset_match(path):
        """True if a derived path belongs to dataset_key (or always, if None)."""
        if dataset_key is None:
            return True
        # Runtime-Data/<MODEL>/<DATASET>/...  and  results/<MODEL>__<DATASET>/...
        parts = path.name
        return path.name == dataset_key or path.name.endswith(f"__{dataset_key}")

    for label, rel in DERIVED_LOCATIONS:
        target = REPO_ROOT / rel
        if not target.exists():
            continue
        if dataset_key is None:
            leftovers = [x for x in target.iterdir() if x.name != "README.md"]
        elif rel == "Runtime-Data":
            # Runtime-Data/<MODEL>/<DATASET>/ - descend one level to the dataset dir
            leftovers = [d for model_dir in target.iterdir()
                         if model_dir.is_dir() and model_dir.name != "README.md"
                         for d in model_dir.iterdir()
                         if d.name == dataset_key]
        else:
            # Step-N results/<MODEL>__<DATASET>/
            leftovers = [x for x in target.iterdir()
                         if x.name != "README.md" and _dataset_match(x)]
        if leftovers:
            found = True
            for item in leftovers:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            scope = dataset_key or "all"
            log("CLEAN", f"reset {label} ({scope})")
    scope = f"for {dataset_key}" if dataset_key else "(all datasets)"
    log("CLEAN", f"system reset complete {scope}"
        if found else f"no stale derived data found {scope}")
    return found


def main(datasets, limit, rebuild=False, shard_index=None, num_shards=None):
    sharded = bool(num_shards and num_shards > 1)
    # decide what actually needs work BEFORE loading any model
    todo = []
    for dataset_key in datasets:
        cfg = load_generator_config(dataset_key)
        out_csv, _ = _shard_paths(dataset_key, shard_index, num_shards)
        if out_csv.exists() and not rebuild:
            what = f"shard {shard_index + 1}/{num_shards}" if sharded else "bank"
            log("SKIP", f"{what} exists (protected - use --rebuild to regenerate)",
                dataset=dataset_key)
        else:
            todo.append((dataset_key, cfg))
    if not todo:
        log("BANK", "nothing to do - all requested outputs exist")
        return

    if sharded:
        # Sharded runs NEVER reset derived data: many array tasks build the same
        # dataset at once and the reset is a single global operation. Each shard
        # writes its own paraphrase_bank.partIIIofNNN.csv (they never collide);
        # merge_shards.py assembles the complete bank and resets ONCE afterwards.
        log("BANK", f"shard mode (index {shard_index}, {num_shards} shards) - "
            f"generating parts only; run merge_shards.py when all shards finish")
    else:
        # New banks are about to be built -> reset the derived pipeline first.
        # Full run (main-setup, all datasets) wipes everything; a single-dataset
        # run (parallel sbatch) wipes ONLY its own dataset, so concurrent jobs on
        # other datasets are never disturbed.
        single = todo[0][0] if len(todo) == 1 else None
        banner("CLEAN", f"new paraphrase bank(s) needed - resetting "
               f"{'dataset ' + single if single else 'the system'} before generating")
        reset_derived_data(dataset_key=single)
        log("BANK", "reset done - now generating paraphrases")

    device = resolve_device("paraphrase bank generation")
    generator = verifier = None
    for dataset_key, cfg in todo:
        torch.manual_seed(cfg["seed"])          # reproducible retry-round sampling
        np.random.seed(cfg["seed"])
        if generator is None:            # shared across datasets
            generator = ParaphraseGenerator(cfg, device)
            verifier = EntailmentVerifier(cfg, device)
        generate_for_dataset(dataset_key, generator, verifier, cfg, limit,
                             rebuild=rebuild, shard_index=shard_index,
                             num_shards=num_shards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of the global DATASETS "
                             "registry (default: all of it)")
    parser.add_argument("--rebuild", action="store_true",
                        help="regenerate an existing bank (previous bank is "
                             "preserved as paraphrase_bank.backup.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="pool hypotheses per dataset (quick trial run)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="split each dataset's pool into this many disjoint "
                             "shards for parallel SLURM array tasks. Each shard "
                             "writes paraphrase_bank.partIIIofNNN.csv; run "
                             "merge_shards.py afterwards to assemble the bank.")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="0-based index of THIS shard (requires --num-shards)")
    args = parser.parse_args()
    if (args.num_shards is None) != (args.shard_index is None):
        parser.error("--num-shards and --shard-index must be given together")
    if args.num_shards is not None:
        if args.num_shards < 1:
            parser.error("--num-shards must be >= 1")
        if not (0 <= args.shard_index < args.num_shards):
            parser.error("--shard-index must be in [0, --num-shards)")
    main(args.datasets.split(",") if args.datasets else list(DATASETS),
         args.limit, rebuild=args.rebuild,
         shard_index=args.shard_index, num_shards=args.num_shards)