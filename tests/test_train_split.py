import numpy as np

from footballsim.vocab import Vocab
from footballsim import config as C
from footballsim.train import split_matches


def test_split_is_disjoint_and_deterministic():
    ids = list(range(100))
    a = split_matches(ids, seed=1)
    b = split_matches(ids, seed=1)
    assert a == b
    tr, va, te = map(set, a)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(ids)


def test_vocab_roundtrip_and_unk(tmp_path):
    v = Vocab.build([["Pass_ok_z1"] * 6, ["Carry_na_z2"] * 6, ["Rare_na_z0"]])
    assert "Rare_na_z0" not in v.stoi  # below min count
    enc = v.encode(["Pass_ok_z1", "Rare_na_z0"])
    assert enc[0] == C.BOS_ID and enc[2] == C.UNK_ID and enc[3] == C.EOS_ID
    assert v.decode(enc) == ["Pass_ok_z1", "<unk>"]
    v.save(tmp_path / "v.json")
    assert Vocab.load(tmp_path / "v.json").itos == v.itos


def test_vocab_built_from_train_only(tiny_ckpt, toy_seqs):
    train_ids = set(tiny_ckpt["split"]["train"])
    train_tokens = {t for toks in toy_seqs[toy_seqs.match_id.isin(train_ids)].toks for t in toks}
    assert set(tiny_ckpt["vocab"][4:]) <= train_tokens
