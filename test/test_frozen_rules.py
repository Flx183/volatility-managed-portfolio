"""Every pre-registered rule table still holds the values it was fixed with (code-review finding 13).

Each fingerprint below was computed from the table as it stood at the commit of the phase that fixed it
(src.frozen.fingerprint, run on a checkout of that commit). A failure here means a frozen rule changed:
restore it, or record the change in a dated PREREGISTRATION entry and build a new table beside it."""

import copy
import importlib
import pickle

import pandas as pd
import pytest

from src.engine.data import DAY, MONTH, WEEK
from src.frozen import Frozen, fingerprint, freeze

PINNED = {
    # 6c887e2: Phase 4 (Entry 1's record of Phases 2-4)
    "src.portfolio.METHODS": ("6c887e2", "8f7d7e8ea649de9f"),
    "src.portfolio.PORTFOLIO_RULES": ("6c887e2", "e19d1a3f2dcf1f44"),
    "src.portfolio.PRIMARY": ("6c887e2", "b39b2ba5eb5d300d"),
    "src.portfolio.TRACKS": ("6c887e2", "2f54e2834c82407a"),
    "src.screen.ETF_FOR": ("6c887e2", "609a5e35ce1c0696"),
    "src.screen.ETF_REGRESSORS": ("6c887e2", "998a6237d5fec6d9"),
    "src.screen.SCREEN_RULES": ("6c887e2", "bf2b6f7353b1d86f"),
    "src.variance.NW_DAILY": ("6c887e2", "71ee45a3c0db9a98"),
    "src.variance.NW_MONTHLY": ("6c887e2", "e7f6c011776e8db7"),
    "src.variance.RULES": ("6c887e2", "59fc876e9a8aa24f"),
    "src.variance.SUBPERIODS": ("6c887e2", "dc648d206a8e396f"),
    # 26fc71e: Phase 5 (Entries 2-3)
    "src.engine.ENGINE_RULES": ("26fc71e", "d4535134fdcf6b27"),
    "src.engine.TIMING": ("26fc71e", "eb93d0a5eb17dd05"),
    "src.engine.evaluation.TEST_RULES": ("26fc71e", "2d0313a69a9becda"),
    "src.engine.execution.COSTS": ("26fc71e", "c5a430afaaea2849"),
    "src.engine.execution.DAYS_PER_YEAR": ("26fc71e", "d6e5a20b30f87216"),
    "src.engine.execution.ETF_KAPPA": ("26fc71e", "4195ffa66c73bae2"),
    # 856de6e: Phase 6 (Entries 4-5)
    "src.models.MODEL_RULES": ("856de6e", "0e662edcddf327f7"),
    "src.models.ORDER": ("856de6e", "b9f3f22c614505ff"),
    "src.models.boost.EVERY": ("856de6e", "ef2d127de37b942b"),
    "src.models.boost.FOLDS": ("856de6e", "4e07408562bedb8b"),
    "src.models.boost.LEAVES": ("856de6e", "0265f23a5b94b32c"),
    "src.models.boost.SETTINGS": ("856de6e", "6b34ee85ed035d17"),
    "src.models.boost.TREES": ("856de6e", "28ee7c2f4055dd9e"),
    "src.models.filters.LAMBDA": ("856de6e", "65ba1590170cf8b7"),
    "src.models.filters.MAX_PERSISTENCE": ("856de6e", "ffe31ff9c96ce4c1"),
    "src.models.filters.SCALE": ("856de6e", "c32c2bab361d45fb"),
    "src.models.inputs.FLOOR": ("856de6e", "4225f957ad91b58e"),
    "src.models.inputs.H": ("856de6e", "785f3ec7eb32f30b"),
    "src.models.inputs.MIN_MONTHS": ("856de6e", "2abaca4911e68fa9"),
    "src.models.inputs.X_MIN_MONTHS": ("856de6e", "39fa9ec190eee7b6"),
    "src.models.scorecard.SCORE_RULES": ("856de6e", "8046b54bc2ffc825"),
    "src.models.scorecard.SPY_COSTS": ("856de6e", "f5857787ee7381db"),
    # 14b27bd: Phase 7 (Entry 6)
    "src.models.weekly.H": ("14b27bd", "ef2d127de37b942b"),
    "src.walkforward.FAMILY": ("14b27bd", "adee66267bfe3663"),
    "src.walkforward.FREQS": ("14b27bd", "33549aac8073b357"),
    "src.walkforward.PRIMARY": ("14b27bd", "0f0dc4f2ca82d939"),
    "src.walkforward.RULES": ("14b27bd", "3820411e00d4b303"),
    "src.walkforward.WF_RULES": ("14b27bd", "911abc9253160f66"),
    "src.walkforward.runs.DOWN_DAYS": ("14b27bd", "785f3ec7eb32f30b"),
    "src.walkforward.runs.GAMMA": ("14b27bd", "ef2d127de37b942b"),
    "src.walkforward.runs.MIN_PAIRS": ("14b27bd", "7dfeebcef0e62640"),
    "src.walkforward.screened.FACTOR_FILTERS": ("14b27bd", "1e6182a117804534"),
    "src.walkforward.stats.BOOT": ("14b27bd", "e297d2db7b6357b5"),
    "src.walkforward.stats.COMBINATION": ("14b27bd", "c5d0a28b414b7993"),
    "src.walkforward.tracks.ETF_FIRST_MONTH": ("14b27bd", "6b51d431df5d7f14"),
    "src.walkforward.tracks.ETF_MIN_PAIRS": ("14b27bd", "76a50887d8f1c2e9"),
    # 99867ee: Phase 8 (Entry 8)
    "src.robustness.BASES": ("99867ee", "82eae0e3187022bc"),
    "src.robustness.CHECKS": ("99867ee", "f70dd4e02bdb0292"),
    "src.robustness.ROBUST_RULES": ("99867ee", "0387aaa6bf6cf582"),
    "src.robustness.crises.BSC": ("99867ee", "672a6a5d846ed37f"),
    "src.robustness.crises.EPISODES": ("99867ee", "13b9e87ed473ed38"),
    "src.robustness.crises.MOMENTUM_CRASHES": ("99867ee", "3fd7298c5267dd7c"),
}
# Wrapped read-only (the rest sit in files the forecast cache is keyed on: pinned, not wrapped).
READ_ONLY = ["src.screen.SCREEN_RULES", "src.screen.ETF_FOR", "src.portfolio.PORTFOLIO_RULES", "src.portfolio.TRACKS",
             "src.engine.ENGINE_RULES", "src.engine.TIMING", "src.engine.execution.COSTS", "src.engine.execution.ETF_KAPPA",
             "src.engine.evaluation.TEST_RULES", "src.models.MODEL_RULES", "src.models.scorecard.SCORE_RULES",
             "src.walkforward.WF_RULES", "src.walkforward.RULES", "src.walkforward.FAMILY", "src.walkforward.FREQS",
             "src.walkforward.runs.MIN_PAIRS", "src.walkforward.stats.BOOT", "src.walkforward.stats.COMBINATION",
             "src.robustness.ROBUST_RULES", "src.robustness.CHECKS", "src.robustness.BASES",
             "src.robustness.crises.EPISODES", "src.robustness.crises.MOMENTUM_CRASHES", "src.robustness.crises.BSC"]


def _table(name):
    mod, attr = name.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)


@pytest.mark.parametrize("name", PINNED)
def test_a_frozen_rule_table_holds_the_values_it_was_fixed_with(name):
    commit, pinned = PINNED[name]
    assert fingerprint(_table(name)) == pinned, f"{name} differs from its table at {commit}"


@pytest.mark.parametrize("name", READ_ONLY)
def test_a_frozen_rule_table_refuses_every_write(name):
    t = _table(name)
    assert isinstance(t, Frozen)
    key = next(iter(t))
    for write in (lambda: t.__setitem__(key, 0), lambda: t.pop(key), lambda: t.update({key: 0}), lambda: t.clear()):
        with pytest.raises(TypeError, match="frozen"):
            write()


def test_phase_7s_c_t_minimums_stay_monthly_and_weekly_and_phase_8_reads_its_own_daily_rule():
    from src.robustness import ROBUST_RULES, checks
    from src.walkforward import WF_RULES, runs
    assert dict(runs.MIN_PAIRS) == {MONTH: 120, WEEK: 520} and WF_RULES["min_pairs"] is runs.MIN_PAIRS
    assert dict(checks.MIN_PAIRS) == {MONTH: 120, WEEK: 520, DAY: ROBUST_RULES["daily_min_pairs"]}
    assert checks.resolve(checks.Spec(freq=DAY), frozen=lambda *a: None)["min_pairs"] == ROBUST_RULES["daily_min_pairs"]


def test_a_frozen_table_reads_pickles_and_copies_as_a_dict():
    t = freeze({"a": 1, "b": {"c": [1, {"d": 2}]}, "e": (3, {"f": 4})})
    assert isinstance(t["b"], Frozen) and isinstance(t["b"]["c"][1], Frozen) and isinstance(t["e"][1], Frozen)
    assert pickle.loads(pickle.dumps(t)) == t and copy.deepcopy(t) == t
    assert pd.DataFrame({"value": t})["value"].to_dict()["a"] == 1          # the notebooks' way of showing rules
    mutable = {**t, "a": 2}
    assert type(mutable) is dict and mutable["a"] == 2 and t["a"] == 1
    assert fingerprint({"x": 1, "y": [1.0, None]}) == fingerprint({"y": (1.0, None), "x": 1})
    assert fingerprint({"x": 120}) != fingerprint({"x": 120.0})
