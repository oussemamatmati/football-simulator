import pytest

from footballsim import config as C
from footballsim.data import build_possessions, make_token, parse_token, zone_center, zone_of


def test_zone_bounds_and_corners():
    assert zone_of(0, 0) == 0
    assert zone_of(119.9, 79.9) == C.N_ZONES - 1
    # out-of-range coordinates are clipped, not crashed
    assert zone_of(130, 90) == C.N_ZONES - 1
    assert zone_of(-5, -5) == 0


def test_zone_center_roundtrip():
    for z in range(C.N_ZONES):
        assert zone_of(*zone_center(z)) == z


def test_token_roundtrip():
    tok = make_token("Ball Recovery", "na", 13)
    assert tok == "BallRecovery_na_z13"
    assert parse_token(tok) == ("BallRecovery", "na", 13)
    assert parse_token("<eos>") is None


def test_possessions_only_possessing_team_and_length(toy_events, toy_seqs):
    assert len(toy_seqs) > 0
    assert toy_seqs.toks.str.len().between(2, C.MAX_LEN - 2).all()
    for toks in toy_seqs.toks:
        assert all(parse_token(t) is not None for t in toks)


def test_defending_team_events_are_excluded(toy_events):
    ev = toy_events.copy()
    ev["team"] = "Z"  # nobody is the possession team
    assert len(build_possessions(ev)) == 0


@pytest.mark.parametrize("bad", ["Pass", "Pass_ok"])
def test_parse_token_rejects_malformed(bad):
    assert parse_token(bad) is None
