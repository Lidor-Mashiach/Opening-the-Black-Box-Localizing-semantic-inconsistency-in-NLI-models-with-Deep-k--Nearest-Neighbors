"""Model loading, batched prediction, and per-layer representation extraction."""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NUM_LABELS = 3

# Dataset label convention (HuggingFace SNLI / MNLI / ANLI):
CANONICAL_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


def prediction_remap(model):
    """Lookup array remapping the CHECKPOINT's label order to the DATASET's.

    Published NLI checkpoints often use a different id order (e.g. the
    official *-mnli checkpoints use 0=contradiction, 1=neutral,
    2=entailment). The checkpoint's id2label names are matched against the
    canonical convention; returns an int lookup array only when a real remap
    is needed, else None. Checkpoints fine-tuned by this repo save proper
    id2label names (see training.py), so they always map to identity.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    remap = {}
    for idx, name in id2label.items():
        name_lower = str(name).lower()
        for canonical, target in CANONICAL_LABELS.items():
            if canonical in name_lower:
                remap[int(idx)] = target
                break
    if len(remap) == NUM_LABELS and any(remap[i] != i for i in remap):
        lookup = np.arange(NUM_LABELS)
        for src, dst in remap.items():
            lookup[src] = dst
        return lookup
    return None


def get_device(context="model loading"):
    """Central GPU detection - loud + actionable (see common/gpu.py)."""
    from .gpu import resolve_device
    return resolve_device(context)


def ensure_raw_backbone(cfg):
    """Make sure the pretrained backbone is present INSIDE the project, at
    Models/raw/<MODEL>/, and return that local path.

    First call on a machine: download the HF weights (config + tokenizer +
    safetensors) straight into the project folder. Every later call: the
    files are already there, so nothing is downloaded again. The HuggingFace
    home cache is never used - everything the code needs lives in the project
    (heavy weights are kept out of git by .gitignore, not by living outside
    the repo).
    """
    from .config_loader import raw_backbone_dir
    from .logging_utils import log
    local = raw_backbone_dir(cfg)
    marker = local / "config.json"
    if marker.exists():
        log("TRAIN", f"backbone already in the project at {local} - "
            f"no download needed", cfg.get("model_key"), cfg.get("dataset_key"))
        return local
    local.mkdir(parents=True, exist_ok=True)
    log("TRAIN", f"downloading pretrained backbone '{cfg['hf_id']}' from "
        f"HuggingFace into the project -> {local}",
        cfg.get("model_key"), cfg.get("dataset_key"))
    AutoTokenizer.from_pretrained(cfg["hf_id"]).save_pretrained(local)
    AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_id"], num_labels=NUM_LABELS).save_pretrained(local)
    return local


def load_model_and_tokenizer(cfg, checkpoint=None):
    """Load tokenizer + 3-way classification head.

    checkpoint : path to a fine-tuned model directory (Step-1 output).
                 When None, the base backbone is loaded FROM WITHIN THE
                 PROJECT (Models/raw/<MODEL>/), downloading it there once if
                 absent - never from the HF home cache.
    """
    if checkpoint is not None:
        source = str(checkpoint)
    else:
        source = str(ensure_raw_backbone(cfg))
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=NUM_LABELS)
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _batches(n, batch_size):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def _encode_batch(tokenizer, first, second, cfg, device):
    enc = tokenizer(list(first), list(second), truncation=True,
                    max_length=cfg["training"]["max_seq_len"],
                    padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict(model, tokenizer, device, premises, targets, cfg, desc="predict"):
    """Batched argmax predictions for (premise, target) pairs -> np.ndarray[int].

    Predictions are always returned in the DATASET label convention
    (0=entailment, 1=neutral, 2=contradiction); published checkpoints with a
    different id order are remapped automatically via their id2label.
    """
    remap = prediction_remap(model)
    if remap is not None:
        print(f"  [{desc}] remapping checkpoint labels -> dataset convention: "
              f"{ {i: int(remap[i]) for i in range(len(remap))} }")
    bs = cfg["encoding"]["batch_size"]
    n = len(premises)
    preds = []
    for start, end in _batches(n, bs):
        enc = _encode_batch(tokenizer, premises[start:end], targets[start:end], cfg, device)
        logits = model(**enc).logits
        batch_preds = logits.argmax(dim=-1).cpu().numpy()
        if remap is not None:
            batch_preds = remap[batch_preds]
        preds.append(batch_preds)
        if start % (bs * 100) == 0:
            print(f"  [{desc}] {end}/{n}", flush=True)
    return np.concatenate(preds)


def _layer_hidden_states(outputs):
    """Tuple of per-layer hidden states for encoder AND encoder-decoder models.

    Encoder models expose `hidden_states`; BART-style classification models
    expose `decoder_hidden_states` instead (their pooled token lives on the
    decoder side). Index 0 is the embedding layer and is dropped, so entry i
    corresponds to transformer layer i+1.
    """
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        hs = outputs.decoder_hidden_states
    return hs[1:]


@torch.no_grad()
def extract_layer_representations(model, tokenizer, device, premises, targets,
                                  cfg, desc="encode"):
    """Per-layer pooled representation of every (premise, target) pair.

    pooling 'cls' -> first token of every layer (BERT / RoBERTa / DeBERTa).
    pooling 'eos' -> last non-padding token of every layer (BART).

    Returns np.ndarray [n, n_layers, dim] with dtype from cfg[encoding][dtype].
    """
    pooling = cfg["pooling"]
    bs = cfg["encoding"]["batch_size"]
    out_dtype = np.float16 if cfg["encoding"]["dtype"] == "float16" else np.float32
    n = len(premises)
    chunks = []
    for start, end in _batches(n, bs):
        enc = _encode_batch(tokenizer, premises[start:end], targets[start:end], cfg, device)
        outputs = model(**enc, output_hidden_states=True)
        stacked = torch.stack(_layer_hidden_states(outputs), dim=1)  # [b, L, seq, dim]
        if pooling == "cls":
            pooled = stacked[:, :, 0, :]
        elif pooling == "eos":
            last = enc["attention_mask"].sum(dim=1) - 1               # [b]
            idx = last.view(-1, 1, 1, 1).expand(-1, stacked.size(1), 1, stacked.size(3))
            pooled = stacked.gather(2, idx).squeeze(2)
        else:
            raise ValueError(f"unknown pooling: {pooling!r}")
        chunks.append(pooled.cpu().numpy().astype(out_dtype))
        if start % (bs * 50) == 0:
            print(f"  [{desc}] {end}/{n}", flush=True)
    return np.concatenate(chunks)
