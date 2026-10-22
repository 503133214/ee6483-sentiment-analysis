"""CPU-only checks for the GloVe + BiLSTM / TextCNN script (no GloVe download needed).
    python -m tests.test_train_rnn
"""
from __future__ import annotations

import numpy as np
import torch

from src import train_rnn as R


class _KV:  # stand-in for gensim KeyedVectors
    def __init__(self, words, dim=8):
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vector_size = dim
        self._m = np.random.default_rng(0).normal(size=(len(words), dim)).astype(np.float32)

    def __getitem__(self, w):
        return self._m[self.key_to_index[w]]


def test_tokenizer_matches_glove_style():
    assert R.tokenize("I don't like it, but it's OK!") == ["i", "do", "n't", "like", "it", ",", "but", "it", "'s", "ok", "!"]
    assert R.tokenize("can't won't I'm you're we've they'll he'd") == ["ca", "n't", "wo", "n't", "i", "'m", "you", "'re", "we", "'ve", "they", "'ll", "he", "'d"]
    assert R.tokenize("Size 8.5 for $19.99 :)") == ["size", "8", ".", "5", "for", "$", "19", ".", "99", ":", ")"]
    assert R.tokenize("") == [] and R.tokenize("   ") == []


def test_vocab_and_encode():
    kv = _KV(["good", "bad", "the", "n't", "shoes"])
    train = [["the", "shoes", "are", "good"], ["bad", "shoes", "zzz"], ["zzz", "qqq", "n't"]]
    itos, stoi, emb, stats = R.build_vocab(train, kv, min_count=2, pretrained=True, rng=np.random.default_rng(1))
    assert itos[:2] == [R.PAD, R.UNK] and "zzz" in stoi and "qqq" not in stoi and "are" not in stoi  # count<2 and not in GloVe
    assert "n't" in stoi and emb.shape == (len(itos), 8) and torch.all(emb[R.PAD_ID] == 0)
    assert torch.allclose(emb[stoi["good"]], torch.tensor(kv["good"]))
    assert stats["vocab_size"] == len(itos) and 0 < stats["glove_init_fraction"] < 1
    enc, es = R.encode([["good", "qqq"], list("abcdefghij")], stoi, max_len=6, head_frac=0.5)
    assert enc[0] == [stoi["good"], R.UNK_ID] and len(enc[1]) == 6 and es["frac_truncated"] == 0.5
    enc_empty, _ = R.encode([[]], stoi, 6, 0.5)
    assert enc_empty == [[R.UNK_ID]]


def _models(vocab=20, dim=8):
    emb = torch.randn(vocab, dim)
    emb[0] = 0
    return [R.BiLSTMClassifier(emb.clone(), hidden=6, layers=1, dropout=0.0), R.TextCNNClassifier(emb.clone(), filters=4, kernels=[3, 4, 5], dropout=0.0)]


def test_padding_does_not_change_outputs():
    torch.manual_seed(0)
    seq = [3, 7, 2, 9, 11, 4, 5]
    for model in _models():
        model.eval()
        ids_a = torch.tensor([seq + [R.PAD_ID] * 9])          # padded to 16
        mask_a = torch.tensor([[True] * len(seq) + [False] * 9])
        ids_b = torch.tensor([seq + [R.PAD_ID] * 1])           # padded to 8
        mask_b = torch.tensor([[True] * len(seq) + [False] * 1])
        with torch.no_grad():
            assert torch.allclose(model(ids_a, mask_a), model(ids_b, mask_b), atol=1e-5), type(model).__name__


def test_batches_and_predict_order():
    enc = [[1, 2, 3], [1], [1, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    batches = list(R.make_batches([0, 1, 2, 3], enc, [0, 1, 0, 1], batch_size=3, shuffle=False, rng=None))
    assert [list(b[0]) for b in batches] == [[1, 2, 0], [3]]
    ids, mask = batches[0][1], batches[0][2]
    assert ids.shape == (3, R.MIN_BATCH_LEN) and mask.sum(1).tolist() == [1, 2, 3] and batches[1][1].shape[1] == 16
    model = _models()[1]
    probs = R.predict(model, [3, 0, 2], enc, batch_size=2, device=torch.device("cpu"))
    single = [R.predict(model, [i], enc, 1, torch.device("cpu"))[0] for i in (3, 0, 2)]
    assert np.allclose(probs, single, atol=1e-6)  # order of `indices` preserved despite length sorting


def test_partial_guard_and_run_id():
    try:
        R.parse_args(["--folds", "0"])
    except SystemExit:
        pass
    else:
        raise AssertionError("partial run must be refused without --no-save/--allow-partial")
    a = R.parse_args(["--arch", "textcnn", "--folds", "1,2", "--limit", "32", "--allow-partial"])
    assert R.run_id_for(a) == "glove_textcnn_f128k345_len160_lr0.001_cwsqrt_seed42_f1-2_n32" and a.dropout == 0.5
    b = R.parse_args(["--no-pretrained"])
    assert b.name.startswith("rand_bilstm_h256x1") and b.dropout == 0.3


def test_textcnn_short_sequences_are_finite():
    model = _models()[1]
    model.eval()
    ids = torch.tensor([[3, R.PAD_ID, R.PAD_ID, R.PAD_ID, R.PAD_ID, R.PAD_ID, R.PAD_ID, R.PAD_ID], [3, 4, 5, 6, 7, 8, 9, 2]])
    mask = torch.tensor([[True] + [False] * 7, [True] * 8])
    with torch.no_grad():
        out = model(ids, mask)
    assert torch.isfinite(out).all() and out.abs().max() < 50, out  # a 1-token text gives zero conv features, not -1e4


def test_train_fold_smoke_and_early_stop():
    import numpy as np
    from src.data import TEXT_COL, load_test, load_train_folds
    df = load_train_folds()
    test = load_test()
    tokens_train = [R.tokenize(t) for t in df[TEXT_COL]]
    tokens_test = [R.tokenize(t) for t in test[TEXT_COL]]
    kv = _KV(["the", "shoes", "great", "not", "n't", "size", "fit"], dim=8)
    args = R.parse_args(["--device", "cpu", "--folds", "0", "--limit", "96", "--epochs", "4", "--patience", "1", "--freeze-emb-epochs", "99", "--hidden", "4", "--no-save"])
    r = R.train_fold(0, args, df, tokens_train, tokens_test, kv, torch.device("cpu"), lambda *a: None)
    assert len(r["oof_probs"]) == len(r["va_idx"]) and len(r["test_probs"]) == len(r["te_idx"]) == 96
    assert r["epochs_run"] <= 4 and r["n_params_trainable"] == r["n_params_no_emb"] and not r["emb_trained"]  # frozen for the whole run
    assert r["vocab"]["vocab_size"] >= 2 and 0 <= r["vocab"]["unk_rate_val"] <= 1


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
