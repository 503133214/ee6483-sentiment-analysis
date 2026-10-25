"""CPU-only checks for the PLM fine-tuning helpers (tokenisers must be in the HF cache).
    python -m tests.test_train_plm
"""
from __future__ import annotations

import numpy as np
from transformers import AutoTokenizer

from src import train_plm as T
from src.data import LABEL_COL, load_train_folds

MODELS = ["distilbert-base-uncased", "roberta-base"]


def test_special_affixes_and_truncation():
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        prefix, suffix = T.special_affixes(tok)
        assert len(prefix) == 1 and len(suffix) == 1, (name, prefix, suffix)
        assert tok("x", add_special_tokens=True)["input_ids"][0] == prefix[0]
        texts = ["short one", " ".join(["word"] * 400), "", "!!! ???", "café ½ size"]
        enc, stats = T.encode_texts(tok, texts, max_len=32, truncation="head_tail", head_frac=0.75)
        assert all(len(e) <= 32 for e in enc)
        assert enc[0][0] == prefix[0] and enc[0][-1] == suffix[0]
        assert len(enc[1]) == 32 and stats["n_truncated"] == 1
        budget = 30
        head = int(budget * 0.75)
        raw = tok(texts[1], add_special_tokens=False)["input_ids"]
        assert enc[1][1:1 + head] == raw[:head] and enc[1][1 + head:-1] == raw[len(raw) - (budget - head):]
        assert enc[2] == prefix + suffix  # empty text -> specials only
        enc_head, _ = T.encode_texts(tok, texts, max_len=32, truncation="head", head_frac=0.75)
        assert enc_head[1][1:-1] == raw[:budget]


def test_fold_indices_and_run_id():
    df = load_train_folds()
    args = T.parse_args(["--folds", "0,1", "--limit", "64", "--allow-partial", "--name", "x"])
    tr, va, te = T.fold_indices(0, args, df, 1851)
    y = df[LABEL_COL].to_numpy()
    assert len(tr) == 64 and len(te) == 64 and set(y[va]) == {0, 1}
    assert (df["fold"].to_numpy()[va] == 0).all() and (df["fold"].to_numpy()[tr] != 0).all()
    assert T.run_id_for(args) == "x_seed42_f0-1_n64"
    full = T.parse_args(["--name", "x"])
    assert T.run_id_for(full) == "x_seed42" and not full.partial
    tr, va, te = T.fold_indices(3, full, df, 1851)
    assert len(tr) + len(va) == len(df) and len(te) == 1851
    assert T.config_signature(full) == T.config_signature(T.parse_args(["--name", "y", "--folds", "2", "--no-save"]))
    assert T.config_signature(full) != T.config_signature(T.parse_args(["--name", "x", "--lr", "1e-5"]))
    default = T.parse_args([])
    assert default.name == "distilbert_base_uncased_len128_bs32_ep3_lr2e-05_cwsqrt"


def test_partial_guard_plain():
    try:
        T.parse_args(["--folds", "0"])
    except SystemExit:
        pass
    else:
        raise AssertionError("partial run without --no-save/--allow-partial must be refused")
    try:
        T.parse_args(["--folds", "0,0", "--no-save"])
    except SystemExit:
        pass
    else:
        raise AssertionError("duplicate fold ids must be refused")


def test_class_weights_and_batches():
    y = np.array([1] * 85 + [0] * 15)
    w = T.class_weights(y, "sqrt").numpy()
    assert abs(w[0] / w[1] - np.sqrt(85 / 15)) < 1e-6
    w = T.class_weights(y, "balanced").numpy()
    assert abs(w[0] / w[1] - 85 / 15) < 1e-6
    assert T.class_weights(y, "none") is None
    try:
        T.class_weights(np.array([1, 1, 1]), "sqrt")
    except AssertionError:
        pass
    else:
        raise AssertionError("single-class subset must be refused")
    enc = [[1, 2, 3], [1], [1, 2], [1, 2, 3, 4]]
    batches = list(T.make_batches([0, 1, 2, 3], enc, [0, 1, 0, 1], batch_size=3, pad_id=9, shuffle=False, rng=None))
    assert [list(b[0]) for b in batches] == [[1, 2, 0], [3]]  # length-sorted
    ids, mask = batches[0][1], batches[0][2]
    assert ids.shape == (3, T.PAD_MULTIPLE) and ids[:, :3].tolist() == [[1, 9, 9], [1, 2, 9], [1, 2, 3]]
    assert (ids[:, 3:] == 9).all() and mask.sum(1).tolist() == [1, 2, 3]
    assert batches[0][3].tolist() == [1, 0, 0]


def test_lora_trainable_subset_and_pad_token():
    import torch
    from transformers import AutoTokenizer
    args = T.parse_args(["--model", "distilbert-base-uncased", "--lora", "--lora-r", "4", "--no-save", "--folds", "0", "--limit", "8"])
    tok = AutoTokenizer.from_pretrained(args.model)
    model = T.load_model(args, tok)
    total = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert 0 < trainable < 0.05 * total, (trainable, total)
    assert any("lora" in n for n in names) and any("classifier" in n or "pre_classifier" in n or "score" in n for n in names), names
    assert model.config.pad_token_id == tok.pad_token_id
    assert args.name.endswith("_lora4")
    assert T.LORA_TARGETS["qwen"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_frozen_pooling_padding_invariant():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from src import train_frozen as FZ
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    enc, _ = T.encode_texts(tok, ["a short one", "this one is quite a bit longer than the other sentence here"], 32, "head_tail", 0.75)
    for pooling in ("mean", "last", "cls"):
        together = FZ.extract(model, enc, tok.pad_token_id, 2, torch.device("cpu"), pooling, torch.float32, lambda *a: None)
        alone = [FZ.extract(model, [e], tok.pad_token_id, 1, torch.device("cpu"), pooling, torch.float32, lambda *a: None)[0] for e in enc]
        assert all(abs(together[i] - alone[i]).max() < 1e-4 for i in range(2)), pooling


if __name__ == "__main__":
    import inspect, sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
