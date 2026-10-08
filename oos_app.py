"""
oos_app.py

Out-of-sample (OOS) performance test as ONE self-contained Streamlit file.
It needs only streamlit, pandas and numpy (Streamlit installs all three), so no
other file and no extra package is required.

Users upload (1) weight recommendations and (2) a price history. The page
computes realized returns after each recommendation date and reports risk and
return statistics, charts and data checks. Nothing is stored; uploads live in
the memory of the session.

Run locally from an Anaconda prompt:
    streamlit run oos_app.py

Structure of this file:
  1. Reading uploaded files (delimiter and decimal mark are detected)
  2. Calculation functions (weights, prices, backtest, statistics)
  3. Streamlit page
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# 1. Reading uploaded files
# ---------------------------------------------------------------------------
SEPARATORS = {"Auto-detect": None, "Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t"}
DECIMALS = ["Auto", ".", ","]

_COMMA_NUMBER = re.compile(r"^-?\d+,\d+$")


def detect_separator(data: bytes) -> str:
    """Pick the delimiter that occurs most often in the header line."""
    lines = data[:20000].decode("utf-8-sig", errors="ignore").splitlines()
    header = lines[0] if lines else ""
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _has_decimal_comma(frame: pd.DataFrame, min_share: float = 0.8, sample: int = 200) -> bool:
    """True if a text column mostly holds numbers written like 12,5."""
    for col in frame.columns:
        if pd.api.types.is_string_dtype(frame[col]) or frame[col].dtype == object:
            values = frame[col].dropna().astype(str).head(sample)
            if len(values) and values.str.match(_COMMA_NUMBER).mean() >= min_share:
                return True
    return False


def read_table(data: bytes, separator=None, decimal: str = "Auto") -> pd.DataFrame:
    """
    Read a delimited file from bytes (UTF-8 first, then Latin-1).
    separator None = detect; decimal "Auto" = detect, otherwise "." or ",".
    """
    sep = separator if separator is not None else detect_separator(data)
    error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), sep=sep, decimal="." if decimal == "Auto" else decimal,
                                encoding=encoding)
            if decimal == "Auto" and _has_decimal_comma(frame):
                frame = pd.read_csv(io.BytesIO(data), sep=sep, decimal=",", encoding=encoding)
            return frame
        except UnicodeDecodeError as exc:
            error = exc
    raise error


def guess_column(columns, candidates):
    """First column whose lower-case name appears in the candidate list (or None)."""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


# ---------------------------------------------------------------------------
# Parameters (all defaults can be changed from the app sidebar)
# ---------------------------------------------------------------------------
@dataclass
class OosParams:
    # Trading days between the last price date on or before the recommendation
    # date and the purchase at the close. 1 = buy at the close of the next day.
    exec_lag: int = 1
    # "buy_and_hold": weights drift between recommendation dates.
    # "constant_mix": weights are reset to the target every period.
    holding_mode: str = "buy_and_hold"
    # What happens to weight on assets without a usable price:
    # "cash" (kept as cash with 0 % return) or "renormalize" (spread over the rest).
    unmatched_mode: str = "cash"
    # Transaction cost in basis points of traded notional.
    cost_bps: float = 0.0
    # Recommendation dates before this date are ignored (None = use all).
    oos_start: pd.Timestamp | None = None
    # Last return day that is evaluated (None = last price date).
    oos_end: pd.Timestamp | None = None
    # A recommendation is skipped if the purchase date lies more than this many
    # calendar days after the recommendation date.
    max_base_gap_days: int = 7


# Metric labels with their display type. The app uses these for formatting.
METRIC_KINDS = {
    "First return day": "date",
    "Last return day": "date",
    "Observations": "int",
    "Cumulative return": "pct",
    "Annualized return": "pct",
    "Annualized volatility": "pct",
    "Sharpe ratio": "ratio",
    "Sortino ratio": "ratio",
    "Omega ratio": "ratio",
    "Maximum drawdown": "pct",
    "Calmar ratio": "ratio",
    "VaR 95% per period (historical)": "pct",
    "CVaR 95% per period (historical)": "pct",
    "Share of positive periods": "pct",
    "Best period": "pct",
    "Worst period": "pct",
}
BENCH_KINDS = {
    "Benchmark cumulative return": "pct",
    "Excess cumulative return": "pct",
    "Tracking error (annualized)": "pct",
    "Information ratio": "ratio",
    "Beta": "ratio",
    "Jensen's alpha (annualized)": "pct",
    "Treynor ratio (annualized)": "pct",
}


# ---------------------------------------------------------------------------
# 1. Weights
# ---------------------------------------------------------------------------
def clean_weights(raw, date_col, strategy_col, asset_col, weight_col,
                  est_end_col=None, source_name="", zero_tol=1e-12,
                  dayfirst=False):
    """
    Return (tidy, diagnostics, n_bad_rows).

    tidy has the columns date, strategy, asset, weight, est_end.
    Weights are rescaled so that they sum to 1 for every (date, strategy).
    diagnostics lists the weight sum before rescaling and the number of
    positions per (date, strategy).
    """
    # Step 1: build a table with standard column names
    strategy = (raw[strategy_col].astype(str).str.strip() if strategy_col
                else pd.Series(source_name, index=raw.index))
    est_end = (pd.to_datetime(raw[est_end_col], errors="coerce", dayfirst=dayfirst).dt.normalize()
               if est_end_col else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]"))
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst).dt.normalize(),
        "strategy": strategy,
        "asset": raw[asset_col].astype(str).str.strip(),
        "weight": pd.to_numeric(raw[weight_col], errors="coerce"),
        "est_end": est_end,
    })

    # Step 2: drop rows that cannot be used and count them
    bad = df[["date", "asset", "weight"]].isna().any(axis=1) | (df["asset"] == "")
    n_bad = int(bad.sum())
    df = df.loc[~bad]

    # Step 3: collapse duplicate (date, strategy, asset) rows by summing weights
    keys = ["date", "strategy", "asset"]
    df = df.groupby(keys, as_index=False, sort=True).agg(
        weight=("weight", "sum"), est_end=("est_end", "max"))

    # Step 4: remove numerical residuals (for example -1e-17) and zero weights
    df.loc[df["weight"].abs() <= zero_tol, "weight"] = 0.0
    df = df.loc[df["weight"] != 0.0].copy()

    # Step 5: rescale to a sum of 1 per (date, strategy) and keep the old sum
    grp = df.groupby(["date", "strategy"])["weight"]
    sums = grp.transform("sum")
    diag = (df.assign(sum_before=sums)
              .groupby(["date", "strategy"], as_index=False)
              .agg(sum_before=("sum_before", "first"), positions=("asset", "size")))
    df["weight"] = np.where(sums != 0, df["weight"] / sums, np.nan)
    df = df.dropna(subset=["weight"]).reset_index(drop=True)
    return df, diag, n_bad


# ---------------------------------------------------------------------------
# 2. Prices
# ---------------------------------------------------------------------------
def prepare_prices(raw, layout, date_col, asset_col=None, price_col=None,
                   ffill_limit=5, dayfirst=False):
    """
    Return (wide, info). wide has one row per date and one column per asset.
    layout is "wide" (one column per asset) or "long" (date, asset, price).
    """
    # Step 1: convert the upload to a wide table
    if layout == "wide":
        dates = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst)
        wide = raw.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
        wide.index = pd.DatetimeIndex(dates).normalize()
    else:
        long = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst).dt.normalize(),
            "asset": raw[asset_col].astype(str).str.strip(),
            "price": pd.to_numeric(raw[price_col], errors="coerce"),
        }).dropna()
        wide = long.pivot_table(index="date", columns="asset", values="price", aggfunc="last")

    # Step 2: tidy the index and the columns
    wide = wide.loc[~wide.index.isna()]
    wide = wide.loc[~wide.index.duplicated(keep="last")].sort_index()
    wide.columns = [str(c).strip() for c in wide.columns]
    wide = wide.loc[:, ~pd.Index(wide.columns).duplicated()]

    # Step 3: prices must be positive numbers
    n_invalid = int((wide <= 0).sum().sum())
    wide = wide.where(wide > 0)
    wide = wide.dropna(axis=1, how="all")

    # Step 4: carry the last price forward over short gaps (holidays, missing cells)
    if ffill_limit and ffill_limit > 0:
        wide = wide.ffill(limit=int(ffill_limit))

    info = {
        "dates": len(wide),
        "assets": wide.shape[1],
        "first": wide.index.min() if len(wide) else pd.NaT,
        "last": wide.index.max() if len(wide) else pd.NaT,
        "invalid_cells": n_invalid,
    }
    return wide, info


# ---------------------------------------------------------------------------
# 3. Matching asset names
# ---------------------------------------------------------------------------
def _strip_suffix(name):
    """AAPL.O -> AAPL (text after the last dot is treated as exchange suffix)."""
    return str(name).strip().upper().rsplit(".", 1)[0]


def build_asset_map(weight_assets, price_columns, ignore_suffix=False, explicit=None):
    """
    Map every asset name in the weight files to a price column (or None).
    Order of rules: explicit mapping table, exact match ignoring case,
    optional match ignoring the exchange suffix (only if the result is unique).
    """
    explicit = explicit or {}
    exact = {str(c).strip().upper(): c for c in price_columns}
    by_stem = {}
    if ignore_suffix:
        groups = {}
        for c in price_columns:
            groups.setdefault(_strip_suffix(c), []).append(c)
        by_stem = {k: v[0] for k, v in groups.items() if len(v) == 1}

    mapping = {}
    for a in weight_assets:
        key = str(a).strip().upper()
        if key in explicit and explicit[key] in price_columns:
            mapping[a] = explicit[key]
        elif key in exact:
            mapping[a] = exact[key]
        elif ignore_suffix and _strip_suffix(a) in by_stem:
            mapping[a] = by_stem[_strip_suffix(a)]
        else:
            mapping[a] = None
    return mapping


# ---------------------------------------------------------------------------
# 4. Out-of-sample backtest
# ---------------------------------------------------------------------------
PERIOD_COLUMNS = [
    "Recommendation date", "Purchase date", "First return day", "Last return day",
    "Holdings used", "Weight matched", "Unmatched holdings", "Carried-forward prices",
    "One-way turnover", "Cost", "Period return", "Estimation window check", "Status",
]


def _skipped_row(rec_date, reason):
    row = {c: np.nan for c in PERIOD_COLUMNS}
    row["Recommendation date"] = rec_date
    row["Status"] = "skipped: " + reason
    return row


def run_strategy_backtest(w, prices, params):
    """
    Evaluate the weights of ONE strategy against the price table.

    w needs the columns date, col (price column or None), weight, est_end.
    Returns (daily_returns, periods).

    Timing: the last price date on or before the recommendation date is the
    signal day s. The portfolio is bought at the close of day s + exec_lag.
    Returns are earned from the next price date on, until the close of the
    purchase day of the next recommendation (or the last evaluated day).
    """
    idx = prices.index
    n = len(idx)
    last_idx = n - 1
    if params.oos_end is not None:
        last_idx = min(last_idx, int(idx.searchsorted(params.oos_end, side="right")) - 1)

    # Step 1: find the purchase day of every recommendation date
    plan, skipped = [], []
    rec_dates = sorted(w["date"].unique())
    if params.oos_start is not None:
        rec_dates = [d for d in rec_dates if d >= params.oos_start]
    for d in rec_dates:
        signal = int(idx.searchsorted(d, side="right")) - 1
        buy = signal + int(params.exec_lag)
        if buy < 0 or buy + 1 > last_idx:
            skipped.append(_skipped_row(d, "no price data after the purchase date"))
        elif (idx[buy] - d).days > params.max_base_gap_days:
            skipped.append(_skipped_row(d, "purchase date too far after recommendation date"))
        else:
            plan.append((d, buy))

    # Step 2: if two recommendations map to the same purchase day, keep the later one
    dedup = {}
    for d, buy in plan:
        if buy in dedup:
            skipped.append(_skipped_row(dedup[buy], "replaced by a later recommendation on the same day"))
        dedup[buy] = d
    plan = [(d, b) for b, d in sorted(dedup.items())]

    pieces, rows = [], []
    prev_end = {}            # drifted weights at the end of the previous period

    # Step 3: evaluate every holding period
    for k, (d, buy) in enumerate(plan):
        seg_end = plan[k + 1][1] if k + 1 < len(plan) else last_idx
        seg_end = min(seg_end, last_idx)

        # 3a: weights of this recommendation, merged onto price columns
        rows_d = w.loc[w["date"] == d]
        mapped = (rows_d.dropna(subset=["col"]).groupby("col", as_index=False)["weight"].sum())
        base_px = prices.iloc[buy].reindex(mapped["col"]).to_numpy()
        usable = mapped.loc[np.isfinite(base_px)]
        total = float(rows_d["weight"].sum())
        usable_sum = float(usable["weight"].sum())
        coverage = usable_sum / total if total else np.nan
        n_unmatched = int(len(rows_d) - len(usable))

        # 3b: decide how unmatched weight is treated
        wi = usable["weight"].to_numpy(dtype=float)
        if params.unmatched_mode == "renormalize" and usable_sum != 0:
            wi = wi * (total / usable_sum)
            cash = 0.0
        else:
            cash = total - usable_sum
        cols = list(usable["col"])
        if not cols:
            skipped.append(_skipped_row(d, "no holding has a price on the purchase date"))
            continue

        # 3c: relative price path since the purchase day
        px = prices.iloc[buy:seg_end + 1][cols]
        carried = int(px.isna().sum().sum())
        growth = (px / px.iloc[0]).ffill().to_numpy(dtype=float)

        # 3d: daily portfolio returns
        if params.holding_mode == "constant_mix":
            asset_ret = growth[1:] / growth[:-1] - 1.0
            r = asset_ret @ wi
            end_w = dict(zip(cols, wi))
        else:
            value = cash + growth @ wi
            bad = np.where(value <= 0)[0]
            if bad.size:                      # only possible with short positions
                value = value[:bad[0]]        # cut the period before the value reaches zero
            r = value[1:] / value[:-1] - 1.0
            end_w = dict(zip(cols, wi * growth[len(value) - 1] / value[-1]))
        if len(r) == 0:
            skipped.append(_skipped_row(d, "no valid return days"))
            continue

        # 3e: turnover and cost at the purchase
        union = set(end_w) | set(prev_end) | set(cols)
        new_w = dict(zip(cols, wi))
        delta = sum(abs(new_w.get(c, 0.0) - prev_end.get(c, 0.0)) for c in union)
        cost = params.cost_bps / 1e4 * delta
        r[0] = (1.0 + r[0]) * (1.0 - cost) - 1.0
        prev_end = end_w

        # 3f: store the daily returns and the period summary
        dates_r = idx[buy + 1: buy + 1 + len(r)]
        pieces.append(pd.Series(r, index=dates_r))
        est_end = rows_d["est_end"].max()
        if pd.isna(est_end):
            check = "n/a"
        else:
            check = "OK" if dates_r[0] > est_end else "FAIL: returns inside estimation window"
        rows.append({
            "Recommendation date": d,
            "Purchase date": idx[buy],
            "First return day": dates_r[0],
            "Last return day": dates_r[-1],
            "Holdings used": len(cols),
            "Weight matched": coverage,
            "Unmatched holdings": n_unmatched,
            "Carried-forward prices": carried,
            "One-way turnover": delta / 2.0,
            "Cost": cost,
            "Period return": float(np.prod(1.0 + r) - 1.0),
            "Estimation window check": check,
            "Status": "ok" if len(dates_r) == seg_end - buy else "ok (cut)",
        })

    returns = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    periods = pd.DataFrame(rows + skipped, columns=PERIOD_COLUMNS)
    if len(periods):
        periods = periods.sort_values("Recommendation date").reset_index(drop=True)
    return returns, periods


def run_oos_test(weights, prices, asset_map, params):
    """
    Run the test for every strategy in the tidy weight table.
    Returns (returns_by_strategy, periods).
    """
    weights = weights.copy()
    weights["col"] = weights["asset"].map(asset_map)
    series, periods = {}, []
    for strat, grp in weights.groupby("strategy", sort=False):
        ret, per = run_strategy_backtest(grp, prices, params)
        per.insert(0, "Strategy", strat)
        periods.append(per)
        if len(ret):
            series[strat] = ret
    returns = pd.DataFrame(series).sort_index() if series else pd.DataFrame()
    periods = pd.concat(periods, ignore_index=True) if periods else pd.DataFrame()
    return returns, periods


def benchmark_returns(prices, column):
    """Simple period returns of one price column."""
    return prices[column].pct_change()


# ---------------------------------------------------------------------------
# 5. Performance statistics
# ---------------------------------------------------------------------------
def infer_periods_per_year(index):
    """Guess the return frequency from the median gap between dates."""
    if len(index) < 3:
        return 252
    gap = pd.Series(index).diff().dt.days.median()
    if gap <= 4:
        return 252
    if gap <= 8:
        return 52
    if gap <= 35:
        return 12
    if gap <= 100:
        return 4
    return 1


def _div(a, b):
    return a / b if (b is not None and np.isfinite(b) and b != 0) else np.nan


def drawdown_series(ret):
    """Drawdown from the running peak of the wealth index (start value 1)."""
    wealth = (1.0 + ret).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return wealth / peak - 1.0


def performance_metrics(ret, ppy, rf_annual=0.0):
    """Return a dict with the entries of METRIC_KINDS."""
    r = ret.dropna().astype(float)
    out = {k: np.nan for k in METRIC_KINDS}
    n = len(r)
    if n == 0:
        return out
    out["First return day"], out["Last return day"], out["Observations"] = r.index[0], r.index[-1], n
    cum = float(np.prod(1.0 + r.to_numpy()) - 1.0)
    out["Cumulative return"] = cum
    out["Best period"], out["Worst period"] = float(r.max()), float(r.min())
    out["Share of positive periods"] = float((r > 0).mean())
    if n < 2:
        return out

    rf_p = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0        # risk-free rate per period
    ex = r - rf_p
    out["Annualized return"] = (1.0 + cum) ** (ppy / n) - 1.0 if 1.0 + cum > 0 else np.nan
    out["Annualized volatility"] = float(r.std(ddof=1) * np.sqrt(ppy))
    out["Sharpe ratio"] = _div(ex.mean(), ex.std(ddof=1)) * np.sqrt(ppy)
    downside = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2)))
    out["Sortino ratio"] = _div(ex.mean(), downside) * np.sqrt(ppy)
    out["Omega ratio"] = _div(np.maximum(ex, 0).sum(), np.maximum(-ex, 0).sum())
    mdd = float(drawdown_series(r).min())
    out["Maximum drawdown"] = mdd
    out["Calmar ratio"] = _div(out["Annualized return"], abs(mdd))
    var = float(np.percentile(r, 5))
    out["VaR 95% per period (historical)"] = var
    out["CVaR 95% per period (historical)"] = float(r[r <= var].mean())
    return out


def benchmark_metrics(ret, bench, ppy, rf_annual=0.0):
    """Benchmark-relative statistics on the dates both series share."""
    out = {k: np.nan for k in BENCH_KINDS}
    pair = pd.concat([ret.rename("r"), bench.rename("b")], axis=1, join="inner").dropna()
    if len(pair) < 2:
        return out
    r, b = pair["r"], pair["b"]
    rf_p = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0
    cum_r = float(np.prod(1.0 + r.to_numpy()) - 1.0)
    cum_b = float(np.prod(1.0 + b.to_numpy()) - 1.0)
    diff = r - b
    te = float(diff.std(ddof=1) * np.sqrt(ppy))
    beta = _div(r.cov(b), b.var(ddof=1))
    excess_r, excess_b = r - rf_p, b - rf_p
    out["Benchmark cumulative return"] = cum_b
    out["Excess cumulative return"] = cum_r - cum_b
    out["Tracking error (annualized)"] = te
    out["Information ratio"] = _div(diff.mean() * ppy, te)
    out["Beta"] = beta
    out["Jensen's alpha (annualized)"] = (excess_r.mean() - beta * excess_b.mean()) * ppy
    out["Treynor ratio (annualized)"] = _div(excess_r.mean() * ppy, beta)
    return out


def monthly_table(ret):
    """Compounded returns by calendar month, plus a total per year."""
    r = ret.dropna()
    if r.empty:
        return pd.DataFrame()
    months = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = months.unstack()
    names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    table = table.rename(columns=names)
    table["Year (compounded)"] = (1.0 + r).groupby(r.index.year).prod() - 1.0
    table.index.name = "Year"
    return table


# ---------------------------------------------------------------------------
# Configuration: every default below can be changed here or in the sidebar
# ---------------------------------------------------------------------------
CONFIG = {
    "page_title": "Out-of-sample performance test",
    "intro": ("Upload portfolio recommendations and a price history. The page measures what "
              "each recommendation earned after its date."),
    "disclaimer": ("Results are hypothetical and computed from the uploaded files. They exclude "
                   "taxes, financing and borrowing costs unless entered above, and they do not "
                   "predict future results. This page is informational and is not investment advice."),
    "file_types": ["csv", "txt", "tsv"],
    # Column-name candidates for the automatic column mapping (lower case)
    "date_candidates": ["recommendation_date", "as_of_date", "rebalance_date", "date", "asof", "as_of"],
    "strategy_candidates": ["strategy", "portfolio", "model"],
    "asset_candidates": ["ticker", "asset", "symbol", "security", "ric"],
    "weight_candidates": ["weight", "weights", "allocation", "w"],
    "est_end_candidates": ["estimation_end"],
    "price_candidates": ["adj close", "adj_close", "adjusted_close", "close", "price", "last"],
    # Defaults of the sidebar controls
    "zero_tol": 1e-12,
    "ffill_limit": 5,
    "ignore_suffix": False,
    "exec_lag": 1,
    "max_base_gap_days": 7,
    "cost_bps": 0.0,
    "rf_annual_pct": 0.0,
    "periods_per_year_options": [252, 52, 12, 4, 1],
    "holding_modes": {"Buy and hold (weights drift)": "buy_and_hold",
                      "Constant mix (weights reset every day)": "constant_mix"},
    "unmatched_modes": {"Keep as cash (0 % return)": "cash",
                        "Spread over matched assets": "renormalize"},
    "min_coverage_warning": 0.95,
    "default_strategies_in_chart": 6,
}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table(data: bytes, separator, decimal: str) -> pd.DataFrame:
    """Read an uploaded delimited text file (delimiter and decimal mark are detected by default)."""
    return read_table(data, separator, decimal)


@st.cache_data(show_spinner=False)
def cached_prices(data, separator, decimal, layout, date_col, asset_col, price_col, ffill_limit, dayfirst):
    """Read and prepare the price file once per combination of settings."""
    raw = load_table(data, separator, decimal)
    return prepare_prices(raw, layout, date_col, asset_col, price_col, ffill_limit, dayfirst)


guess = guess_column


def pick(label, columns, guessed, key, optional=False, help_text=None):
    """Selectbox that starts on the guessed column; returns None for '(none)'."""
    options = (["(none)"] if optional else []) + list(columns)
    index = options.index(guessed) if guessed in options else 0
    choice = st.selectbox(label, options, index=index, key=key, help=help_text)
    return None if choice == "(none)" else choice


def as_text_dates(df):
    """Format datetime columns as text so tables show plain dates."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def style_metrics(table, kinds):
    """Format a metrics table according to the display type of each column."""
    fmt = {}
    out = table.copy()
    for col, kind in kinds.items():
        if col not in out.columns:
            continue
        if kind == "date":
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            fmt[col] = {"pct": "{:.2%}", "ratio": "{:.2f}", "int": "{:,.0f}"}[kind]
    return out.style.format(fmt, na_rep="n/a")


def wealth_line(returns, price_index, base=100.0):
    """Wealth index (start = base) with a start point on the purchase date."""
    r = returns.dropna()
    loc = max(int(price_index.searchsorted(r.index[0])) - 1, 0)
    x = [price_index[loc]] + list(r.index)
    y = np.r_[base, base * (1.0 + r).cumprod().to_numpy()]
    return x, y


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title=CONFIG["page_title"], layout="wide")
st.title(CONFIG["page_title"])
st.caption(CONFIG["intro"])

# Sidebar part 1: settings that are needed to read the files
with st.sidebar:
    st.header("File and data settings")
    sep_label = st.selectbox("Delimiter", list(SEPARATORS), index=0,
                             help="Auto-detect works for each file separately.")
    separator = SEPARATORS[sep_label]
    decimal = st.selectbox("Decimal mark", DECIMALS, index=0)
    dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)", value=False)
    zero_tol = st.number_input("Treat weights up to this size as zero", min_value=0.0,
                               value=float(CONFIG["zero_tol"]), format="%.1e", step=1e-12,
                               help="Removes numerical residuals such as -1e-17 from optimizer output.")
    ffill_limit = st.number_input("Carry prices forward over gaps of up to (days)", min_value=0,
                                  value=int(CONFIG["ffill_limit"]), step=1)
    ignore_suffix = st.checkbox("Match tickers ignoring the exchange suffix (AAPL.O = AAPL)",
                                value=CONFIG["ignore_suffix"])

# Step 1 and 2: uploads
st.subheader("1. Recommendations (weights)")
weight_files = st.file_uploader("One or several delimited files with date, strategy, asset and weight",
                                type=CONFIG["file_types"], accept_multiple_files=True, key="weights")
st.subheader("2. Price history")
price_file = st.file_uploader("One file with prices after the recommendation dates",
                              type=CONFIG["file_types"], key="prices")
mapping_file = st.file_uploader("Optional: ticker mapping (column 1 = name in the weight files, "
                                "column 2 = name in the price file)",
                                type=CONFIG["file_types"], key="mapping")

with st.expander("What the files need to contain"):
    st.markdown(
        "- **Weights:** one row per date, strategy and asset, with a weight. Weights are rescaled to "
        "sum to 1 for every date and strategy, so fractions and percentages both work.\n"
        "- **Prices:** closing prices, preferably adjusted for splits and dividends, all in the same "
        "currency. The page does not convert currencies. Either one column per asset (wide) or one "
        "row per date and asset (long).\n"
        "- **Out-of-sample:** only prices after each recommendation date are used. If the weight file "
        "has an estimation end date, the page checks that the evaluated returns start after it.")

if not weight_files or price_file is None:
    st.info("Upload at least one weight file and one price file to start.")
    st.stop()

# Step 3: map the columns of every weight file and clean the weights
frames, diagnostics = [], []
for i, f in enumerate(weight_files):
    raw = load_table(f.getvalue(), separator, decimal)
    cols = list(raw.columns)
    with st.expander(f"Column mapping for {f.name}", expanded=len(weight_files) == 1):
        c1, c2, c3 = st.columns(3)
        with c1:
            date_col = pick("Recommendation date", cols, guess(cols, CONFIG["date_candidates"]), f"date{i}")
            strat_col = pick("Strategy", cols, guess(cols, CONFIG["strategy_candidates"]), f"strat{i}",
                             optional=True, help_text="If none, the file name is used as strategy name.")
        with c2:
            asset_col = pick("Asset", cols, guess(cols, CONFIG["asset_candidates"]), f"asset{i}")
            weight_col = pick("Weight", cols, guess(cols, CONFIG["weight_candidates"]), f"weight{i}")
        with c3:
            est_col = pick("Estimation end date", cols, guess(cols, CONFIG["est_end_candidates"]), f"est{i}",
                           optional=True, help_text="Used only to check that evaluated returns come after it.")
    try:
        tidy, diag, n_bad = clean_weights(raw, date_col, strat_col, asset_col, weight_col, est_col,
                                             source_name=f.name.rsplit(".", 1)[0], zero_tol=zero_tol,
                                             dayfirst=dayfirst)
    except Exception as exc:
        st.error(f"{f.name}: the weight file could not be read with this mapping ({exc}).")
        st.stop()
    if n_bad:
        st.warning(f"{f.name}: {n_bad} rows without a valid date, asset or weight were ignored.")
    tidy["source"] = f.name
    diag["source"] = f.name
    frames.append(tidy)
    diagnostics.append(diag)

weights = pd.concat(frames, ignore_index=True)
if weights.empty:
    st.error("No usable weights were found. Check the column mapping.")
    st.stop()

# Strategy names that occur in more than one file get the file name appended
files_per_strategy = weights.groupby("strategy")["source"].nunique()
clash = files_per_strategy[files_per_strategy > 1].index
mask = weights["strategy"].isin(clash)
weights.loc[mask, "strategy"] = weights.loc[mask, "strategy"] + " [" + weights.loc[mask, "source"] + "]"
weight_diag = pd.concat(diagnostics, ignore_index=True)

# Step 4: map the price file and prepare the price table
raw_prices = load_table(price_file.getvalue(), separator, decimal)
pcols = list(raw_prices.columns)
g_asset = guess(pcols, CONFIG["asset_candidates"])
g_price = guess(pcols, CONFIG["price_candidates"])
with st.expander("Column mapping for the price file", expanded=True):
    layout_label = st.radio("Layout", ["Wide: one column per asset", "Long: one row per date and asset"],
                            index=1 if (g_asset and g_price) else 0, horizontal=True)
    layout = "wide" if layout_label.startswith("Wide") else "long"
    p1, p2, p3 = st.columns(3)
    with p1:
        p_date = pick("Date", pcols, guess(pcols, CONFIG["date_candidates"]), "p_date")
    if layout == "long":
        with p2:
            p_asset = pick("Asset", pcols, g_asset, "p_asset")
        with p3:
            p_price = pick("Price", pcols, g_price, "p_price")
    else:
        p_asset = p_price = None
try:
    prices, price_info = cached_prices(price_file.getvalue(), separator, decimal, layout, p_date,
                                       p_asset, p_price, int(ffill_limit), dayfirst)
except Exception as exc:
    st.error(f"The price file could not be read with this mapping ({exc}).")
    st.stop()
if prices.empty or len(prices) < 3:
    st.error("The price table has fewer than three dates. Check the date column and the layout.")
    st.stop()

# Optional explicit mapping of asset names
explicit = {}
if mapping_file is not None:
    mp = load_table(mapping_file.getvalue(), separator, decimal)
    if mp.shape[1] >= 2:
        explicit = {str(a).strip().upper(): str(b).strip() for a, b in zip(mp.iloc[:, 0], mp.iloc[:, 1])}
    else:
        st.warning("The mapping file needs two columns and was ignored.")

# Sidebar part 2: settings of the test itself
with st.sidebar:
    st.header("Test settings")
    first_rec = weights["date"].min().date()
    oos_start = st.date_input("Ignore recommendations dated before", value=first_rec)
    oos_end = st.date_input("Evaluate returns up to", value=prices.index.max().date(),
                            min_value=prices.index.min().date(), max_value=prices.index.max().date())
    exec_lag = st.number_input("Purchase lag (trading days after the recommendation)", min_value=0,
                               max_value=20, value=int(CONFIG["exec_lag"]), step=1,
                               help="0 buys at the close of the recommendation day, 1 at the close of the next trading day.")
    holding_label = st.radio("Holding rule", list(CONFIG["holding_modes"]), index=0)
    unmatched_label = st.radio("Weight on assets without a price", list(CONFIG["unmatched_modes"]), index=0)
    cost_bps = st.number_input("Transaction cost (basis points of traded value)", min_value=0.0,
                               value=float(CONFIG["cost_bps"]), step=1.0)
    rf_pct = st.number_input("Risk-free rate (% per year)", value=float(CONFIG["rf_annual_pct"]), step=0.1)
    max_gap = st.number_input("Maximum days between recommendation and purchase", min_value=0,
                              value=int(CONFIG["max_base_gap_days"]), step=1)
    ppy_inferred = infer_periods_per_year(prices.index)
    ppy_choice = st.selectbox("Periods per year (for annualizing)", ["Auto"] + CONFIG["periods_per_year_options"],
                              help=f"Auto uses {ppy_inferred}, estimated from the spacing of the price dates.")
    ppy = ppy_inferred if ppy_choice == "Auto" else int(ppy_choice)
    bench_options = ["(none)"] + list(prices.columns)
    bench_col = st.selectbox("Benchmark (a column of the price file)", bench_options, index=0)

params = OosParams(
    exec_lag=int(exec_lag), holding_mode=CONFIG["holding_modes"][holding_label],
    unmatched_mode=CONFIG["unmatched_modes"][unmatched_label], cost_bps=float(cost_bps),
    oos_start=pd.Timestamp(oos_start), oos_end=pd.Timestamp(oos_end), max_base_gap_days=int(max_gap))
rf = float(rf_pct) / 100.0

# Step 5: run the test
asset_map = build_asset_map(weights["asset"].unique(), list(prices.columns), ignore_suffix, explicit)
returns, periods = run_oos_test(weights, prices, asset_map, params)

tab_sum, tab_chart, tab_detail, tab_checks = st.tabs(["Summary", "Charts", "Strategy detail", "Data checks"])

# Warnings that apply to the whole result
if not periods.empty:
    low = periods["Weight matched"].lt(CONFIG["min_coverage_warning"]).sum()
    if low:
        st.warning(f"{int(low)} holding periods have less than {CONFIG['min_coverage_warning']:.0%} of the "
                   "weight matched to a price. See the Data checks tab.")
    fails = periods["Estimation window check"].astype(str).str.startswith("FAIL").sum()
    if fails:
        st.error(f"{int(fails)} holding periods include returns inside the estimation window. These are not "
                 "out-of-sample. Increase the purchase lag or remove those recommendations.")

if returns.empty:
    with tab_sum:
        st.error("No out-of-sample returns could be computed. The most likely reasons are that the price "
                 "file ends before the recommendation dates or that the asset names do not match.")
    with tab_checks:
        st.dataframe(as_text_dates(periods))
    st.stop()

bench_returns = benchmark_returns(prices, bench_col) if bench_col != "(none)" else None

# Statistics for every strategy
stats = {}
for s in returns.columns:
    row = performance_metrics(returns[s], ppy, rf)
    if bench_returns is not None:
        row.update(benchmark_metrics(returns[s], bench_returns, ppy, rf))
    stats[s] = row
kinds = dict(METRIC_KINDS)
if bench_returns is not None:
    kinds.update(BENCH_KINDS)
stats_table = pd.DataFrame(stats).T[list(kinds)]

with tab_sum:
    shortest = int(stats_table["Observations"].min())
    if shortest < ppy:
        st.info(f"The shortest sample has {shortest} return periods, fewer than one year ({ppy}). "
                "Annualized figures scale short samples up and are indicative only.")
    st.dataframe(style_metrics(stats_table, kinds))
    d1, d2 = st.columns(2)
    d1.download_button("Download statistics (CSV)", stats_table.to_csv().encode("utf-8"),
                       file_name="oos_statistics.csv", mime="text/csv")
    d2.download_button("Download returns per period (CSV)", returns.to_csv().encode("utf-8"),
                       file_name="oos_returns.csv", mime="text/csv")

with tab_chart:
    chosen = st.multiselect("Strategies", list(returns.columns),
                            default=list(returns.columns)[:CONFIG["default_strategies_in_chart"]])
    if chosen:
        # Wealth index of every chosen strategy (start = 100), one column per strategy
        wealth = {}
        for s in chosen:
            x, y = wealth_line(returns[s], prices.index)
            wealth[s] = pd.Series(y, index=pd.DatetimeIndex(x))
        if bench_returns is not None:
            first = min(returns[s].dropna().index[0] for s in chosen)
            last = max(returns[s].dropna().index[-1] for s in chosen)
            bx, by = wealth_line(bench_returns.loc[first:last], prices.index)
            wealth[f"Benchmark: {bench_col}"] = pd.Series(by, index=pd.DatetimeIndex(bx))
        st.markdown("**Wealth index (start = 100)**")
        st.line_chart(pd.DataFrame(wealth))

        # Drawdown from the running peak of every chosen strategy
        st.markdown("**Drawdown**")
        st.line_chart(pd.DataFrame({s: drawdown_series(returns[s].dropna()) for s in chosen}))

with tab_detail:
    one = st.selectbox("Strategy", list(returns.columns))
    per = periods.loc[periods["Strategy"] == one].drop(columns="Strategy")
    st.markdown("**Holding periods**")
    st.dataframe(as_text_dates(per).style.format({
        "Weight matched": "{:.1%}", "One-way turnover": "{:.2%}", "Cost": "{:.3%}",
        "Period return": "{:.2%}"}, na_rep=""))
    monthly = monthly_table(returns[one])
    if not monthly.empty:
        st.markdown("**Monthly returns (compounded)**")
        st.dataframe(monthly.style.format("{:.2%}", na_rep=""))
    st.download_button("Download holding periods (CSV)", per.to_csv(index=False).encode("utf-8"),
                       file_name="oos_periods.csv", mime="text/csv")

with tab_checks:
    st.markdown("**Price table**")
    st.write(f"{price_info['dates']} dates from {price_info['first']:%Y-%m-%d} to {price_info['last']:%Y-%m-%d}, "
             f"{price_info['assets']} assets, {price_info['invalid_cells']} non-positive prices removed.")
    st.markdown("**Weight files**")
    wd = weight_diag.copy()
    wd["date"] = wd["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(wd.rename(columns={"sum_before": "Weight sum before rescaling", "positions": "Positions"}))
    unmatched = weights.assign(col=weights["asset"].map(asset_map))
    unmatched = unmatched.loc[unmatched["col"].isna()]
    st.markdown("**Assets without a matching price column**")
    if unmatched.empty:
        st.write("All assets in the weight files were matched.")
    else:
        table = (unmatched.groupby("asset").agg(strategies=("strategy", "nunique"), largest_weight=("weight", "max"))
                 .sort_values("largest_weight", ascending=False))
        st.dataframe(table.style.format({"largest_weight": "{:.2%}"}))
    st.markdown("**All holding periods**")
    st.dataframe(as_text_dates(periods).style.format({
        "Weight matched": "{:.1%}", "One-way turnover": "{:.2%}", "Cost": "{:.3%}",
        "Period return": "{:.2%}"}, na_rep=""))

st.divider()
st.caption(CONFIG["disclaimer"])


# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v2.0 (2026-10-08)  Single-file version. Changes against v1.0:
#   - Added: this docstring and the imports io, re, dataclass at the top.
#   - Moved unchanged into this file: the contents of file_io.py (section 1)
#     and oos_metrics.py (section 2); the "fio." and "om." prefixes were removed.
#   - Removed: "import plotly.graph_objects as go", the FIT width helper and
#     the "import file_io" / "import oos_metrics" lines.
#   - Replaced: the two plotly charts in the Charts tab by st.line_chart
#     (wealth index and drawdown).
#   - Modified: in benchmark_metrics, pd.concat(...) now uses join="inner"
#     (same result as the previous outer join followed by dropna, and no
#     pandas deprecation warning).
