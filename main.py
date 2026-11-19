"""Refresh-aware earners-over-players forecaster for GTA VI PS5 trophy rates.

Model (stdlib only, deterministic, <8 s per case):

Exophase refreshes a game's counts on its own schedule, so a read shows the
rates as of its last refresh, and the tracked-player base grows between
refreshes. A rate is earners / players, so this program forecasts the two
counts separately and divides, instead of extrapolating the rate itself:

  * players N(t) grows as a power law in (1 + hours since release):
    log N = a_N + b_N * log(1 + h).  b_N is fit by OLS over the case's
    *distinct* snapshots (consecutive identical reads collapse to one: a
    flat stretch means "no refresh yet", not "no growth"), shrunk toward a
    prior that decays as the launch surge fades, b_N0 = 1 / (1 + h_ref/72).
  * earners E_i(t) per trophy follow the same log-time law with slope g_i,
    fit per trophy where the history varies, shrunk toward a prior
        g0_i = b_N * tier_factor * (0.10 + 0.45 * sqrt(rate_i / 100)),
    so earners accumulate slowly: on launch night only 4-13 of 638 tracked
    players held each non-intro trophy after ~half a day of play, so new
    players contribute almost no new earners of a rare trophy in their
    first hours while the player base multiplies. Common, early trophies
    (high rate, bronze) track the base more closely; rare / late ones
    (platinum, gold, low rate) grow slowly and dilute. A trophy at 100%
    has g0 = b_N, so it stays at 100%. The prior puts the practice
    median at ~0.25x the last rate (log-midpoint between "earners frozen",
    638/3868 = 0.165, and "earners keep pace", ~0.4), minimising the
    worst-case logit error when the refresh size is unknown.
  * the median rate at the target is 100 * E_i(target) / N(target),
    projected from the last *distinct* snapshot, which correctly bridges a
    stale tail: when the history is flat, the last read's counts are stale
    and the projection covers the whole unrefreshed interval.
  * the spread is a logit-normal fan whose sigma grows with the horizon
    and with staleness (a long flat tail means a pending refresh of unknown
    size) and with small-count noise (early earners are single digits).

TimesFM was checked first (skill check_system.py): this VM has ~1 GB RAM
free, below the 2 GB preflight threshold, so no neural forecaster runs
here; the statistical model above is the entry instead.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of GTA VI: 2026-11-19 (game.release_date in the case file).
RELEASE = "2026-11-19T00:00:00Z"

TIER_FACTOR = {"platinum": 0.70, "gold": 0.85, "silver": 1.00, "bronze": 1.10}


def parse_time(s):
    """ISO-8601 UTC (with optional fractional seconds) -> epoch seconds."""
    s = s.replace("Z", "+00:00")
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; None when xs do not vary."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def shrink(fitted, prior, n_distinct, strength=4.0):
    """Precision-style shrinkage: fitted slope wins as distinct reads grow."""
    if fitted is None:
        return prior
    w = n_distinct / (n_distinct + strength)
    return w * fitted + (1.0 - w) * prior


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    t0 = parse_time(case.get("game", {}).get("release_date", "2026-11-19") + "T00:00:00Z"
                    if len(str(case.get("game", {}).get("release_date", ""))) == 10
                    else RELEASE)
    t_at = parse_time(case["at"])
    t_target = parse_time(case["target"])
    horizon_days = max(0.0, (t_target - t_at) / 86400.0)

    # Collapse consecutive identical reads: one snapshot per refresh.
    snaps = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, parse_time(read["at"]), read))
    n_distinct = len(snaps)
    last_key, last_t, last_read = snaps[-1]
    n_last = float(last_read["players_tracked"])
    h_ref = max(0.0, (last_t - t0) / 3600.0)
    h_tgt = max(h_ref, (t_target - t0) / 3600.0)
    x_ref = math.log1p(h_ref)
    x_tgt = math.log1p(h_tgt)
    dx = x_tgt - x_ref

    # Staleness: hours from the last distinct snapshot to the checkpoint.
    stale_hours = max(0.0, (t_at - last_t) / 3600.0)
    stale = 0.35 if stale_hours > 2.0 else 0.10

    # --- players: log-log slope, shrunk toward the fading-surge prior ---
    b_prior = 1.0 / (1.0 + h_ref / 72.0)
    if n_distinct >= 2:
        xs = [math.log1p(max(0.0, (t - t0) / 3600.0)) for _, t, _ in snaps]
        ys = [math.log(max(1.0, float(r["players_tracked"]))) for _, _, r in snaps]
        b_n = shrink(ols_slope(xs, ys), b_prior, n_distinct)
    else:
        b_n = b_prior
    b_n = min(1.5, max(0.05, b_n))
    n_tgt = n_last * math.exp(b_n * dx)

    sigma_base = math.sqrt(0.22 ** 2 + (0.30 * math.sqrt(horizon_days)) ** 2 + stale ** 2)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Last non-null rate for this trophy (walk back over snapshots).
        rate = None
        e_ref = None
        for _, _, r in reversed(snaps):
            v = r["rates"][i]
            if v is not None:
                rate = float(v)
                e_ref = v / 100.0 * float(r["players_tracked"])
                break
        if rate is None:
            rate, e_ref = 0.0, 0.0
        # Continuity correction: early earners are single digits.
        e_ref_c = e_ref + 0.5
        r_frac = min(1.0, max(0.0, rate / 100.0))

        g_prior = b_n * TIER_FACTOR.get(trophy.get("tier", "bronze"), 1.0) \
            * (0.10 + 0.45 * math.sqrt(r_frac))
        g_prior = min(1.05 * b_n, max(0.05 * b_n, g_prior))
        if r_frac >= 0.999:
            g_prior = b_n  # a 100% trophy's earners are the player base

        if n_distinct >= 3:
            xs, ys = [], []
            for _, t, r in snaps:
                v = r["rates"][i]
                if v is None:
                    continue
                xs.append(math.log1p(max(0.0, (t - t0) / 3600.0)))
                ys.append(math.log(v / 100.0 * float(r["players_tracked"]) + 0.5))
            g_i = shrink(ols_slope(xs, ys), g_prior, len(xs))
        else:
            g_i = g_prior
        g_i = min(1.5 * b_n, max(0.0, g_i))

        e_tgt = e_ref_c * math.exp(g_i * dx)
        median = 100.0 * e_tgt / n_tgt
        median = min(99.99, max(0.01, median))

        # Extra spread for tiny earner counts (Poisson-ish, in logit units).
        sigma = math.sqrt(sigma_base ** 2 + min(0.25, 0.4 / math.sqrt(e_ref_c)) ** 2)
        lg = logit(median)
        qs = [round4(min(100.0, max(0.0, inv_logit(lg + z * sigma)))) for z in Z]
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
