"""Zero-shot sentiment classification with an instruction-tuned LLM (no training), scored under the shared protocol.

    python -m src.llm_zero_shot --model Qwen/Qwen3.8-27B --quant 4bit --name qwen3.8_27b_zeroshot
    python -m src.llm_zero_shot --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --limit 32 --no-save   # smoke test

Each review is wrapped in a fixed prompt through the model's chat template (thinking disabled);
one forward pass gives the next-token logits at the answer position, and the probability of
"positive" vs "negative" is the softmax over the log-probabilities of the two label words
(summed over surface forms: "positive", " positive", "Positive", ...). The model never sees a
label, so every training review is scored "out of fold" trivially and the same scores serve as
OOF; the test set is scored once. Threshold 0.5 is the pure zero-shot operating point; the
OOF-chosen threshold uses training labels only for calibration (noted in the summary).
Outputs mirror the other scripts (results/oof, test_probs, summary, runs.csv; feature_format
"llm_zero_shot", selection "none").
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src import data as _data
from src.data import FOLD_SEED, LABEL_COL, N_SPLITS, TEXT_COL, load_test, load_train_folds
from src.metrics import STD_DDOF, compute_metrics
from src.train_plm import TORCH_DTYPES, pick_device, seed_everything
from src.utils.resources import Timer, machine_tag, peak_rss_mb
from src.utils.results import compute_run_stats, log_run_stats, runs_row, write_run

transformers.logging.set_verbosity_error()
RESULTS_DIR = Path(os.environ.get("EE6483_RESULTS_DIR", _data.RESULTS_DIR))
NON_CONFIG_KEYS = {"name", "no_save", "allow_partial", "folds", "device", "fold_list", "partial", "scores_cache"}

SYSTEM = "You are a sentiment classifier for online product reviews."
USER = ("Review:\n\"\"\"\n{review}\n\"\"\"\n\n"
        "Is the overall sentiment of this review positive or negative? Answer with exactly one word: positive or negative.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none", help="bitsandbytes weight quantisation (4bit = nf4 + double quant, bf16 compute)")
    p.add_argument("--load-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--max-len", type=int, default=256, help="token budget for the review text itself (head+tail truncation) before it is put into the prompt")
    p.add_argument("--head-frac", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--labels", default="negative,positive", help="label words for class 0 and class 1")
    p.add_argument("--enable-thinking", action="store_true", help="leave the chat template's thinking mode on (default: off, we read the first answer token)")
    p.add_argument("--device", default="auto")
    p.add_argument("--folds", default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--acc-slack", type=float, default=0.003)
    p.add_argument("--scores-cache", default=None, help="optional .npz to checkpoint/restore raw scores (resume after a dropped session)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)
    if args.name is None:
        args.name = f"{args.model.split('/')[-1].replace('-', '_')}_zeroshot" + (f"_{args.quant}" if args.quant != "none" else "")
    folds = list(range(N_SPLITS)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    args.fold_list = folds
    args.partial = len(folds) < N_SPLITS or bool(args.limit)
    if args.partial and not args.no_save and not args.allow_partial:
        p.error("a --folds subset or --limit run is partial: add --no-save or --allow-partial")
    return args


def run_id_for(args) -> str:
    rid = f"{args.name}_seed{args.seed}"
    if len(args.fold_list) < N_SPLITS:
        rid += "_f" + "-".join(map(str, args.fold_list))
    if args.limit:
        rid += f"_n{args.limit}"
    return rid


def truncate_review(tokenizer, text: str, max_len: int, head_frac: float) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_len:
        return text
    head = int(max_len * head_frac)
    return tokenizer.decode(ids[:head]) + " ... " + tokenizer.decode(ids[len(ids) - (max_len - head):])


def build_prompts(tokenizer, texts, args) -> list[str]:
    out = []
    for t in texts:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(review=truncate_review(tokenizer, t, args.max_len, args.head_frac))}]
        try:
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
        except TypeError:  # template without a thinking switch
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(s)
    return out


def label_token_ids(tokenizer, word: str) -> list[int]:
    """First-token ids of the label word's surface forms (with/without leading space, capitalised)."""
    ids = set()
    for form in {word, word.capitalize(), " " + word, " " + word.capitalize()}:
        toks = tokenizer(form, add_special_tokens=False)["input_ids"]
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def load_llm(args, device):
    cfg = AutoConfig.from_pretrained(args.model)
    kwargs = {}
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if args.quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": 0}
    dtype = TORCH_DTYPES[args.load_dtype]
    loader = AutoModelForCausalLM
    if getattr(cfg, "vision_config", None) is not None:  # multimodal checkpoint (Qwen3.5 / Qwen3.8): load the full model, use it text-only
        from transformers import AutoModelForImageTextToText
        loader = AutoModelForImageTextToText
    try:
        model = loader.from_pretrained(args.model, dtype=dtype, **kwargs)
    except TypeError:
        model = loader.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    if args.quant == "none":
        model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score(model, tokenizer, prompts, pos_ids, neg_ids, batch_size, device, log, cache_path=None, cached=None):
    """log p(positive) - log p(negative) at the answer position, per prompt; checkpointed to cache_path."""
    n = len(prompts)
    out = np.full(n, np.nan) if cached is None else cached.copy()
    order = np.argsort([len(p) for p in prompts], kind="stable")  # length-sorted batches
    todo = [i for i in order if np.isnan(out[i])]
    t0, done = time.perf_counter(), 0
    for s in range(0, len(todo), batch_size):
        idx = todo[s:s + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :].float()           # left padding -> last position is the answer position for every row
        lp = torch.log_softmax(logits, dim=-1)
        lp_pos = torch.logsumexp(lp[:, pos_ids], dim=-1)
        lp_neg = torch.logsumexp(lp[:, neg_ids], dim=-1)
        out[idx] = (lp_pos - lp_neg).cpu().numpy()
        done += len(idx)
        if cache_path is not None and (s // batch_size) % 20 == 19:
            np.savez(cache_path, scores=out)
        if (s // batch_size) % 25 == 0:
            el = time.perf_counter() - t0
            log(f"  scored {done}/{len(todo)}  ({el/60:.1f} min, {done/max(el,1e-9)*60:.0f} texts/min)")
    if cache_path is not None:
        np.savez(cache_path, scores=out)
    return out


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    log = functools.partial(print, flush=True)
    run_id = run_id_for(args)
    seed_everything(args.seed)
    t_start = time.perf_counter()
    log(f"zero-shot {args.model} | quant {args.quant} | device {device} | thinking {'on' if args.enable_thinking else 'off'} | run {run_id}{' [PARTIAL]' if args.partial else ''}")

    df = load_train_folds()
    test = load_test()
    y = df[LABEL_COL].to_numpy()
    n_tr = args.limit or len(df)
    n_te = args.limit or len(test)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    neg_word, pos_word = args.labels.split(",")
    pos_ids, neg_ids = label_token_ids(tokenizer, pos_word), label_token_ids(tokenizer, neg_word)
    assert pos_ids and neg_ids and not set(pos_ids) & set(neg_ids), (pos_ids, neg_ids)
    with Timer() as t_prompt:
        prompts = build_prompts(tokenizer, list(df[TEXT_COL][:n_tr]) + list(test[TEXT_COL][:n_te]), args)
    plen = np.array([len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts[:200]])
    log(f"prompts built in {t_prompt.seconds:.1f}s | prompt tokens (first 200): median {np.median(plen):.0f}, max {plen.max()} | label ids pos {pos_ids} neg {neg_ids}")
    log("example prompt:\n" + prompts[0][-600:])

    with Timer() as t_load:
        model = load_llm(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model loaded in {t_load.seconds/60:.1f} min | {n_params/1e9:.2f}B params (quantised modules count their packed size) | GPU mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB" if device.type == "cuda" else f"model loaded in {t_load.seconds:.1f}s")
    cache = Path(args.scores_cache) if args.scores_cache else None
    cached = np.load(cache)["scores"] if (cache and cache.exists()) else None
    if cached is not None:
        log(f"resuming: {int((~np.isnan(cached)).sum())}/{len(cached)} scores cached")
    with Timer() as t_score:
        raw = score(model, tokenizer, prompts, pos_ids, neg_ids, args.batch_size, device, log, cache, cached)
    prob = 1.0 / (1.0 + np.exp(-raw))                              # sigmoid of the log-odds between the two label words
    prob_tr, prob_te = prob[:n_tr], prob[n_tr:]
    log(f"scored {len(prompts)} texts in {t_score.seconds/60:.1f} min ({len(prompts)/max(t_score.seconds,1e-9)*60:.0f} texts/min)")

    folds = args.fold_list
    keep = np.isin(df["fold"].to_numpy()[:n_tr], folds)
    oof = np.full(len(df), np.nan); oof[:n_tr][keep] = prob_tr[keep]
    test_prob = np.zeros(len(test)); test_prob[:n_te] = prob_te
    te_seen = np.zeros(len(test), dtype=bool); te_seen[:n_te] = True
    fold_rows = []
    for k in folds:
        m = (df["fold"].to_numpy()[:n_tr] == k)
        if m.sum() == 0:
            continue
        r = compute_metrics(y[:n_tr][m], prob_tr[m], 0.5); r.update(fold=k, n_val=int(m.sum())); fold_rows.append(r)
    st = compute_run_stats(y, oof, oof.copy(), df["fold"].to_numpy(), fold_rows, [dict(r) for r in fold_rows], args.acc_slack)
    test_pred_pos_ratio = float((test_prob[te_seen] >= st["threshold"]).mean())
    log_run_stats(log, st, len(folds), args.partial, test_pred_pos_ratio, float(y.mean()))
    log(f"== zero-shot @0.5 (no labels used anywhere): test positive ratio {(prob_te >= 0.5).mean():.4f}")
    total = time.perf_counter() - t_start
    if args.no_save:
        log("(--no-save: nothing written)")
        return True

    per_1000 = t_score.seconds / len(prompts) * 1000
    config = {k2: v for k2, v in vars(args).items() if k2 not in NON_CONFIG_KEYS}
    resources = {"feature_dim": None, "n_params": 0, "n_params_total_loaded": n_params, "prompt_build_s": t_prompt.seconds, "load_s": t_load.seconds, "score_s_total": t_score.seconds,
                 "score_s_per_1000_texts": per_1000, "train_s_total": 0.0, "infer_test_s": per_1000 * len(test) / 1000, "wall_s": total, "peak_rss_mb": peak_rss_mb(),
                 "device_mem_mb_max": float(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None, "machine": machine_tag(), "device": str(device), "precision": f"{args.quant if args.quant != 'none' else args.load_dtype}"}
    summary = {"run_id": run_id, "name": args.name, "model": f"{args.model} zero-shot", "feature_format": "llm_zero_shot", "config": {k2: v for k2, v in vars(args).items() if k2 != "fold_list"},
               "protocol": {"dedup": True, "n_train": int(len(df)), "n_splits": N_SPLITS, "folds_run": folds, "partial": args.partial, "fold_seed": FOLD_SEED, "std_ddof": STD_DDOF,
                            "threshold_rule": f"max macro-F1 s.t. acc >= best-acc - {args.acc_slack}; chosen on all training-set scores (calibration only; 0.5 is the pure zero-shot point)",
                            "model_selection": "none (no training; every training review is scored once by the frozen LLM, so OOF == all rows; *_last identical)",
                            "prompt": {"system": SYSTEM, "user": USER, "thinking": args.enable_thinking, "labels": args.labels, "label_token_ids": {"pos": pos_ids, "neg": neg_ids}}},
               "tokenization": {"review_token_budget": args.max_len, "truncation": "head_tail (decoded back to text)"},
               "folds": fold_rows, "folds_last": fold_rows, "zero_shot_at_0.5": compute_metrics(y[:n_tr][keep], prob_tr[keep], 0.5), "test_pred_pos_ratio": test_pred_pos_ratio,
               "test_pred_pos_ratio_at_0.5": float((prob_te >= 0.5).mean()), "resources": resources}
    row = runs_row(run_id, summary["model"], "llm_zero_shot", args.seed, len(folds), args.partial, None, 0, 0.0, resources["infer_test_s"], peak_rss_mb(),
                   f"{device.type}/{resources['precision']}", None if device.type != "cuda" else round(resources["device_mem_mb_max"]), "none", st, test_pred_pos_ratio, config)
    write_run(RESULTS_DIR, run_id, oof, oof.copy(), test_prob, st, [], summary, row, log)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
