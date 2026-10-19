"""
Frozen rule tables
==================
A rule table fixed in PREREGISTRATION is read-only in code, and its fingerprint is pinned in
test/test_frozen_rules.py at the commit of the phase that fixed it.

    freeze(table)        the table with every dict in it made Frozen: reading works as for a dict,
                         any write raises. Lists stay lists (pandas reads a tuple of column names as
                         one key)
    fingerprint(table)   a hash of the table's values: dict keys in any order, functions by name,
                         numbers exactly (120 and 120.0 differ)

To change a frozen rule, write a dated PREREGISTRATION entry and build a new table beside the old
one, never an edit of it: MIN_PAIRS = freeze({**WF_RULES["min_pairs"], DAY: 2520}).

Tables in the files the forecast cache is keyed on (variance.py, models/inputs.py, filters.py,
har.py, boost.py, weekly.py, variations.py) are pinned by fingerprint but not wrapped: any edit to
those files starts a new cache folder and voids every cached forecast.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np


class Frozen(dict):
    """A dict that refuses every write. Copies ({**t}, dict(t), t.copy()) are ordinary dicts."""

    def _refuse(self, *args, **kwargs):
        raise TypeError("a frozen rule table cannot be changed: build a new one, e.g. freeze({**table, key: value})")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self):                       # pickle and deepcopy rebuild it without __setitem__
        return Frozen, (dict(self),)


def freeze(table):
    if isinstance(table, Frozen):                 # already frozen: the same table, not a copy
        return table
    if isinstance(table, dict):
        return Frozen({k: freeze(v) for k, v in table.items()})
    if isinstance(table, list):
        return [freeze(v) for v in table]
    if isinstance(table, tuple):
        return tuple(freeze(v) for v in table)
    return table


def _canonical(x):
    if isinstance(x, dict):
        return {"__dict__": sorted([str(k), _canonical(v)] for k, v in x.items())}
    if isinstance(x, (list, tuple)):
        return [_canonical(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"__dataclass__": type(x).__qualname__, **{f.name: _canonical(getattr(x, f.name)) for f in dataclasses.fields(x)}}
    if callable(x):
        return f"<function {x.__module__}.{x.__qualname__}>"
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return {"__float__": repr(x)}
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return {"__repr__": repr(x)}


def fingerprint(table) -> str:
    return hashlib.sha256(json.dumps(_canonical(table), sort_keys=True).encode()).hexdigest()[:16]
