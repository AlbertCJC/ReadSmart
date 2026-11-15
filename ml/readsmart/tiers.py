"""Tier step: maps the analyzer's speech metrics to Low / Moderate / High using written rules.

The rules ARE the tier criteria (Objective 4). They live in a separate YAML file that the
researchers write with the adviser; this code never sets a threshold itself. While the file has no
version or no rules, every result says "criteria not set" and gives no tier.

File format (criteria/tier_criteria.yaml):
    version: "criteria v1, 2027-01-20"     # required
    approved_by: "..."                       # required
    default_tier: Low                        # tier when no rule matches (required)
    rules:                                   # checked top to bottom; first match wins
      - tier: High
        any:                                 # 'any' = at least one; 'all' = every condition
          - {metric: clarity_score, op: "<=", value: <number>}
          - all:
              - {metric: pauses_per_min, op: ">=", value: <number>}
              - {metric: fillers_per_min, op: ">=", value: <number>}
      - tier: Moderate
        any: [...]
"""
from __future__ import annotations

import operator
from pathlib import Path

import yaml

from .analyzer import METRIC_NAMES

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}


class CriteriaError(ValueError):
    pass


def load_criteria(path, tiers: list[str]) -> dict | None:
    """Returns the criteria dict, or None if the file is still empty (criteria not set)."""
    path = Path(path)
    if not path.exists():
        return None
    c = yaml.safe_load(path.read_text()) or {}
    if not c.get("version") or not c.get("rules"):
        return None
    for key in ("approved_by", "default_tier"):
        if not c.get(key):
            raise CriteriaError(f"criteria file needs '{key}'")
    if c["default_tier"] not in tiers:
        raise CriteriaError(f"default_tier must be one of {tiers}")
    for i, r in enumerate(c["rules"]):
        if r.get("tier") not in tiers:
            raise CriteriaError(f"rule {i}: tier must be one of {tiers}")
        if ("any" in r) == ("all" in r):
            raise CriteriaError(f"rule {i}: use exactly one of 'any' or 'all'")
        _check_cond({k: r[k] for k in ("any", "all") if k in r}, i)
    return c


def _check_cond(c, i):
    if "any" in c or "all" in c:
        for sub in c.get("any", c.get("all")):
            _check_cond(sub, i)
        return
    if c.get("metric") not in METRIC_NAMES:
        raise CriteriaError(f"rule {i}: unknown metric {c.get('metric')}; choose from {METRIC_NAMES}")
    if c.get("op") not in OPS:
        raise CriteriaError(f"rule {i}: op must be one of {list(OPS)}")
    if not isinstance(c.get("value"), (int, float)):
        raise CriteriaError(f"rule {i}: value must be a number")


def _eval(c, m) -> bool:
    if "any" in c:
        return any(_eval(s, m) for s in c["any"])
    if "all" in c:
        return all(_eval(s, m) for s in c["all"])
    return OPS[c["op"]](m[c["metric"]], c["value"])


def assign_tier(metrics: dict, criteria: dict | None) -> dict:
    if criteria is None:
        return {"tier": None, "criteria_version": None, "status": "criteria not set",
                "matched_rule": None}
    for i, r in enumerate(criteria["rules"]):
        if _eval({k: r[k] for k in ("any", "all") if k in r}, metrics):
            return {"tier": r["tier"], "criteria_version": criteria["version"], "status": "ok", "matched_rule": i}
    return {"tier": criteria["default_tier"], "criteria_version": criteria["version"], "status": "ok",
            "matched_rule": None}
