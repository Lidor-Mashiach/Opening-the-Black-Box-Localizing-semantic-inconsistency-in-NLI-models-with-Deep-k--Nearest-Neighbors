"""Generate + verify hypothesis paraphrases for every (model, dataset) pair.

Implements the acquisition protocol documented in Datasets/*/paraphrases/README.md,
mirroring the anchor paper (Arakelyan et al., EACL 2024):

1. SAMPLE   deterministic_bank_subset -> deterministic_eval_sample of the
            model's filtered train_correct.csv, so every sampled hypothesis is
            GUARANTEED to be inside the encoded DkNN bank (Step-3 alignment
            depends on that).
2. GENERATE a local seq2seq paraphraser (paraphrases.generator_model, default:
            humarin/chatgpt_paraphraser_on_T5_base - the leading open
            paraphraser on the HF Hub) produces candidates_per_hypothesis
            rewordings via diverse beam search (decoding per its model card).
3. VERIFY   the strongest public NLI model (paraphrases.verifier_model,
            default: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
            keeps a candidate ONLY if
                hypothesis -> candidate  AND  candidate -> hypothesis
            are BOTH classified 'entailment' (symmetric equivalence).
            This is the base condition that guarantees
            Premise:Hypothesis == Premise:Paraphrase in the ground truth.
4. SAVE     the first per_hypothesis verified candidates per hypothesis ->
            Datasets/<dataset>/paraphrases/<MODEL>__paraphrases.csv
            (+ a stats json). Step-2 consumes this file as-is.

Run:  python generate_paraphrases.py                       (all combinations)
      python generate_paraphrases.py --models BERT-base --datasets SNLI
      python generate_paraphrases.py --limit 50            (quick trial run)

GPU strongly recommended (generation + verification are transformer passes).
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "Research-Pipeline")]

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from common.config_loader import (filtered_dir, list_dataset_keys,
                                  list_model_keys, load_config,
                                  paraphrases_csv)
from common.sampling import (deterministic_bank_subset,
                             deterministic_eval_sample)


def _normalize(text):
    """Comparison key for duplicate detection (case / spacing / final period)."""
    return " ".join(str(text).lower().split()).rstrip(" .?!")


class ParaphraseGenerator:
    """Diverse-beam-search paraphrase candidates from a seq2seq model."""

    def __init__(self, cfg, device):
        model_id = cfg["paraphrases"]["generator_model"]
        print(f"[generator] loading {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
        self.prefix = cfg["paraphrases"]["generator_prompt_prefix"]
        self.n_candidates = cfg["paraphrases"]["candidates_per_hypothesis"]
        self.device = device

    @torch.no_grad()
    def generate(self, hypotheses):
        """list[str] -> list[list[str]] of n_candidates rewordings each.

        Decoding parameters follow the generator's model card (diverse beam
        search: one beam per group, high repetition/diversity penalties).
        """
        inputs = [self.prefix + h for h in hypotheses]
        enc = self.tokenizer(inputs, padding=True, truncation=True,
                             max_length=128, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            num_beams=self.n_candidates,
            num_beam_groups=self.n_candidates,
            num_return_sequences=self.n_candidates,
            diversity_penalty=3.0,
            repetition_penalty=10.0,
            no_repeat_ngram_size=2,
            max_length=128,
        )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        n = self.n_candidates
        return [decoded[i * n:(i + 1) * n] for i in range(len(hypotheses))]


class EntailmentVerifier:
    """Symmetric-equivalence gate: both directions must be 'entailment'."""

    def __init__(self, cfg, device):
        model_id = cfg["paraphrases"]["verifier_model"]
        print(f"[verifier] loading {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        self.batch_size = cfg["encoding"]["batch_size"]
        self.device = device
        self.entail_idx = self._entailment_index()
        print(f"[verifier] entailment index = {self.entail_idx}")

    def _entailment_index(self):
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, name in id2label.items():
            if "entail" in str(name).lower():
                return int(idx)
        raise ValueError("verifier id2label does not name an 'entailment' class - "
                         "choose a verifier with proper label names")

    @torch.no_grad()
    def _predicts_entailment(self, first, second):
        """Boolean array: argmax == entailment for each (first[i], second[i])."""
        flags = []
        for start in range(0, len(first), self.batch_size):
            end = min(start + self.batch_size, len(first))
            enc = self.tokenizer(list(first[start:end]), list(second[start:end]),
                                 padding=True, truncation=True, max_length=128,
                                 return_tensors="pt").to(self.device)
            preds = self.model(**enc).logits.argmax(dim=-1).cpu().numpy()
            flags.append(preds == self.entail_idx)
        return np.concatenate(flags)

    def both_entail(self, hypotheses, candidates):
        forward = self._predicts_entailment(hypotheses, candidates)
        backward = self._predicts_entailment(candidates, hypotheses)
        return forward & backward


def generate_for_combo(model_key, dataset_key, generator, verifier, cfg, limit=None):
    csv_path = filtered_dir(cfg) / "train_correct.csv"
    if not csv_path.exists():
        print(f"[paraphrases] SKIP {model_key} x {dataset_key}: missing {csv_path}\n"
              f"              (run Step-1 build_filtered_dataset.py first)")
        return False

    df = pd.read_csv(csv_path)
    bank_df, _ = deterministic_bank_subset(df, cfg)
    sample = deterministic_eval_sample(bank_df, cfg)
    if limit is not None:
        sample = sample.head(limit)
    per_hyp = cfg["paraphrases"]["per_hypothesis"]
    gen_bs = cfg["paraphrases"]["generation_batch_size"]
    print(f"[paraphrases] {model_key} x {dataset_key}: {len(sample)} hypotheses, "
          f"target {per_hyp} verified paraphrases each")

    rows, candidates_total, verified_total = [], 0, 0
    quota_counts = {"full": 0, "partial": 0, "dropped": 0}
    records = sample.to_dict("records")
    for start in range(0, len(records), gen_bs):
        batch = records[start:start + gen_bs]
        hyps = [r["hypothesis"] for r in batch]
        all_candidates = generator.generate(hyps)

        # de-duplicate per hypothesis (vs the original and among candidates)
        cleaned = []
        for record, cands in zip(batch, all_candidates):
            seen = {_normalize(record["hypothesis"])}
            kept = []
            for cand in cands:
                cand = cand.strip()
                key = _normalize(cand)
                if cand and key not in seen:
                    seen.add(key)
                    kept.append(cand)
            cleaned.append(kept)
            candidates_total += len(kept)

        # one flat verification pass for the whole generation batch
        flat_h = [record["hypothesis"] for record, cands in zip(batch, cleaned) for _ in cands]
        flat_c = [cand for cands in cleaned for cand in cands]
        ok = verifier.both_entail(flat_h, flat_c) if flat_c else np.array([], dtype=bool)

        pos = 0
        for record, cands in zip(batch, cleaned):
            verified = [c for j, c in enumerate(cands) if ok[pos + j]]
            pos += len(cands)
            verified = verified[:per_hyp]
            verified_total += len(verified)
            if len(verified) == per_hyp:
                quota_counts["full"] += 1
            elif verified:
                quota_counts["partial"] += 1
            else:
                quota_counts["dropped"] += 1
            for para_idx, paraphrase in enumerate(verified):
                rows.append({"pair_id": record["pair_id"],
                             "premise": record["premise"],
                             "hypothesis": record["hypothesis"],
                             "paraphrase": paraphrase,
                             "label": record["label"],
                             "para_idx": para_idx})
        done = min(start + gen_bs, len(records))
        print(f"  [paraphrases] {done}/{len(records)} hypotheses "
              f"({len(rows)} verified paraphrases so far)", flush=True)

    out_csv = paraphrases_csv(cfg)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    stats = {
        "model": model_key,
        "dataset": dataset_key,
        "generator_model": cfg["paraphrases"]["generator_model"],
        "verifier_model": cfg["paraphrases"]["verifier_model"],
        "hypotheses_sampled": int(len(sample)),
        "hypotheses_full_quota": quota_counts["full"],
        "hypotheses_partial_quota": quota_counts["partial"],
        "hypotheses_dropped_no_verified": quota_counts["dropped"],
        "candidates_after_dedup": int(candidates_total),
        "verified_kept": int(len(rows)),
        "verification_acceptance_rate":
            float(verified_total / candidates_total) if candidates_total else 0.0,
        "output_csv": str(out_csv),
    }
    with open(out_csv.parent / f"{model_key}__paraphrases_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[paraphrases] saved {len(rows)} paraphrases -> {out_csv}\n"
          f"              full quota: {quota_counts['full']}, partial: "
          f"{quota_counts['partial']}, dropped: {quota_counts['dropped']}")
    return True


def main(models, datasets, limit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paraphrases] device: {device}"
          + ("" if device.type == "cuda" else "  (GPU strongly recommended)"))
    generator = verifier = None
    for dataset_key in datasets:
        for model_key in models:
            cfg = load_config(model_key, dataset_key)
            if generator is None:                       # shared across combos
                generator = ParaphraseGenerator(cfg, device)
                verifier = EntailmentVerifier(cfg, device)
            generate_for_combo(model_key, dataset_key, generator, verifier, cfg, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=None,
                        help="hypotheses per combination (quick trial run)")
    args = parser.parse_args()
    main(args.models.split(",") if args.models else list_model_keys(),
         args.datasets.split(",") if args.datasets else list_dataset_keys(),
         args.limit)
