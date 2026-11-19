import numpy as np
import pytest
import torch

from footballsim import config as C
from footballsim.evaluate import evaluate, markov_perplexity
from footballsim.model import ModelConfig, SmallGPT
from footballsim.simulate import simulate, summarize, zone_distribution
from footballsim.train import load_checkpoint, save_checkpoint
from footballsim.vocab import Vocab


def _model_vocab(ckpt):
    vocab = Vocab(ckpt["vocab"])
    m = SmallGPT(ModelConfig(**ckpt["model_cfg"]))
    m.load_state_dict(ckpt["model_state"])
    return m.eval(), vocab


def test_model_is_causal():
    torch.manual_seed(0)
    m = SmallGPT(ModelConfig(vocab_size=50, d_model=32, n_heads=2, n_layers=2, dropout=0.0)).eval()
    a = torch.randint(4, 50, (2, 12))
    b = a.clone()
    b[:, 8:] = torch.randint(4, 50, (2, 4))  # change only the future
    with torch.no_grad():
        la, lb = m(a), m(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)


def test_simulate_shapes_seed_and_validity(tiny_ckpt):
    model, vocab = _model_vocab(tiny_ckpt)
    s1 = simulate(model, vocab, 300, seed=3, batch_size=100)
    s2 = simulate(model, vocab, 300, seed=3, batch_size=100)
    assert s1 == s2  # reproducible
    assert 0 < len(s1) <= 300
    assert all(len(s) <= C.MAX_LEN - 1 for s in s1)
    specials = set(C.SPECIAL_TOKENS)
    assert all(t not in specials for s in s1 for t in s)


def test_simulate_batches_of_unequal_length_do_not_crash(tiny_ckpt):
    # regression: batches that finish early used to have different widths and could not be concatenated
    model, vocab = _model_vocab(tiny_ckpt)
    simulate(model, vocab, 1000, batch_size=50, seed=1)


def test_prefix_is_respected_and_validated(tiny_ckpt):
    model, vocab = _model_vocab(tiny_ckpt)
    tok = vocab.itos[4]
    sims = simulate(model, vocab, 100, prefix=[tok], seed=0)
    assert sims and all(s[0] == tok for s in sims)
    with pytest.raises(ValueError):
        simulate(model, vocab, 10, prefix=["Nonexistent_ok_z0"])
    with pytest.raises(ValueError):
        simulate(model, vocab, 10, temperature=0)


def test_summarize_and_zone_distribution():
    seqs = [["Pass_ok_z0", "Shot_Goal_z22"], ["Pass_ok_z1", "Pass_fail_z2"]]
    s = summarize(seqs)
    assert s["shot_rate"] == 0.5 and s["goal_rate"] == 0.5 and s["avg_length"] == 2
    assert s["ends_in_final_third"] == 0.5  # z22 is in x-band 5; z2 is not
    d = zone_distribution(seqs)
    assert abs(d.sum() - 1) < 1e-9 and d.shape == (C.N_ZONES,)


def test_markov_perplexity_sane():
    X = np.array([[1, 4, 5, 2, 0], [1, 4, 5, 2, 0]])
    ppl = markov_perplexity(X, X, V=6)
    assert 1.0 <= ppl < 2.0  # fully deterministic data


def test_checkpoint_roundtrip_and_evaluate(tiny_ckpt, toy_seqs, tmp_path):
    path = save_checkpoint(tiny_ckpt, tmp_path / "m.pt")
    model, vocab, ckpt = load_checkpoint(path)
    rep = evaluate(model, vocab, ckpt, toy_seqs, n_sim=200)
    assert rep["transformer_perplexity"] > 1 and rep["markov_perplexity"] > 1
    assert rep["n_test_matches"] == len(ckpt["split"]["test"])
    assert 0 <= rep["shot_calibration_gap"] <= 1
