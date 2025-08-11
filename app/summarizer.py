# app/summarizer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_string_dtype
import math

def infer_column_role(series: pd.Series) -> str:
    """Infer whether column is categorical, numeric, datetime or id-like."""
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        # heuristics to detect ID-like numeric columns
        uniq = series.nunique(dropna=True)
        if uniq > 0 and uniq / max(len(series.dropna()), 1) > 0.9 and uniq > 50:
            return "id"
        return "numeric"
    if is_string_dtype(series):
        uniq = series.nunique(dropna=True)
        # treat as categorical if low cardinality or looks like category
        if uniq <= 50:
            return "categorical"
        # if most values look like numbers stored as strings, still treat as numeric candidate
        sample = series.dropna().astype(str).head(100).tolist()
        numeric_like = sum(1 for v in sample if _looks_like_number(v))
        if numeric_like / max(len(sample), 1) > 0.8:
            return "numeric"
        return "categorical"
    return "unknown"

def _looks_like_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False

def column_statistics(series: pd.Series) -> Dict[str, Any]:
    """Return stats depending on inferred role."""
    role = infer_column_role(series)
    stats = {"role": role, "non_null_count": int(series.count()), "missing_count": int(series.isna().sum())}
    total = len(series)
    stats["missing_pct"] = round(100.0 * stats["missing_count"] / max(total, 1), 2)

    if role == "numeric":
        # safe numeric conversion when necessary
        s = pd.to_numeric(series, errors="coerce")
        stats.update({
            "mean": None if s.count()==0 else float(s.mean()),
            "median": None if s.count()==0 else float(s.median()),
            "min": None if s.count()==0 else float(s.min()),
            "max": None if s.count()==0 else float(s.max()),
            "std": None if s.count()==0 else float(s.std()),
            "unique": int(s.nunique(dropna=True))
        })
        # top / bottom samples
        if s.count() > 0:
            stats["top_5_values"] = s.dropna().value_counts().head(5).to_dict()
    elif role == "categorical":
        s = series.astype(str).replace("nan", pd.NA)
        stats.update({
            "unique": int(s.nunique(dropna=True)),
            "top_values": s.value_counts(dropna=True).head(6).to_dict()
        })
    elif role == "datetime":
        s = pd.to_datetime(series, errors="coerce")
        stats.update({
            "min": None if s.dropna().empty else str(s.min()),
            "max": None if s.dropna().empty else str(s.max()),
            "unique_dates": int(s.nunique(dropna=True))
        })
    elif role == "id":
        stats.update({
            "unique": int(series.nunique(dropna=True)),
            "sample_ids": series.dropna().astype(str).head(10).tolist()
        })
    else:
        stats.update({"unique": int(series.nunique(dropna=True))})
    return stats

def dataframe_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a structured profile of the DataFrame."""
    profile = {"n_rows": int(df.shape[0]), "n_columns": int(df.shape[1]), "columns": {}}
    for col in df.columns:
        try:
            series = df[col]
            profile["columns"][str(col)] = column_statistics(series)
        except Exception as e:
            profile["columns"][str(col)] = {"error": str(e)}
    # compute numeric correlations (top correlated pairs)
    numeric_cols = [c for c, v in profile["columns"].items() if v.get("role") == "numeric"]
    if len(numeric_cols) >= 2:
        # safe correlation
        corr = df[numeric_cols].corr(method='pearson', min_periods=5)
        # pick top absolute correlations excluding self
        corr_pairs = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i+1:]:
                val = corr.at[a, b]
                if not (pd.isna(val)):
                    corr_pairs.append((a, b, float(val)))
        # sort by absolute strength
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        profile["top_numeric_correlations"] = [{"col_a": a, "col_b": b, "corr": round(c, 3)} for a,b,c in corr_pairs[:10]]
    else:
        profile["top_numeric_correlations"] = []

    # simple trend detection: if there is a datetime and a numeric, try basic period aggregation (quarterly)
    dt_cols = [c for c, v in profile["columns"].items() if v.get("role") == "datetime"]
    num_cols = [c for c, v in profile["columns"].items() if v.get("role") == "numeric"]
    if dt_cols and num_cols:
        try:
            dt = pd.to_datetime(df[dt_cols[0]], errors="coerce")
            df_dt = df.copy()
            df_dt["_dt"] = dt
            df_dt = df_dt.dropna(subset=["_dt"])
            if not df_dt.empty:
                df_dt["_quarter"] = df_dt["_dt"].dt.to_period("Q")
                # compute QoQ pct change for top numeric columns (by variance)
                var_ranked = df_dt[num_cols].var().sort_values(ascending=False).index.tolist()
                qtr_trends = {}
                for col in var_ranked[:3]:  # check top 3 numeric columns
                    q = df_dt.groupby("_quarter")[col].sum().astype(float)
                    if len(q) >= 2:
                        # compute last vs previous percent change
                        last = q.iloc[-1]
                        prev = q.iloc[-2] if len(q) >= 2 else None
                        if prev is not None and prev != 0:
                            pct = (last - prev) / abs(prev) * 100.0
                            qtr_trends[col] = {"last_quarter_total": float(last), "prev_quarter_total": float(prev), "pct_change": round(pct,2)}
                profile["quarterly_trends"] = qtr_trends
        except Exception:
            profile["quarterly_trends"] = {}
    else:
        profile["quarterly_trends"] = {}

    return profile

def profile_to_text(profile: Dict[str, Any], max_lines: int = 200) -> str:
    """Convert structured profile to a compact human-readable text summary suitable for LLM."""
    lines = []
    lines.append(f"Dataset: {profile.get('n_rows')} rows x {profile.get('n_columns')} columns")
    # columns summary
    for col, meta in profile.get("columns", {}).items():
        role = meta.get("role")
        miss = meta.get("missing_pct", 0)
        if role == "numeric":
            lines.append(f"- {col} (numeric) | missing: {miss}% | mean: {_fmt(meta.get('mean'))} | min/max: {_fmt(meta.get('min'))}/{_fmt(meta.get('max'))} | unique: {meta.get('unique')}")
        elif role == "categorical":
            top = list(meta.get("top_values", {}).items())[:5]
            top_s = ", ".join([f"{k}({v})" for k,v in top])
            lines.append(f"- {col} (categorical) | missing: {miss}% | unique: {meta.get('unique')} | top: {top_s}")
        elif role == "datetime":
            lines.append(f"- {col} (datetime) | range: {meta.get('min')} to {meta.get('max')}")
        elif role == "id":
            lines.append(f"- {col} (id-like) | unique ids: {meta.get('unique')}")
        else:
            lines.append(f"- {col} ({role}) | unique: {meta.get('unique')} | missing: {miss}%")
    # correlations
    if profile.get("top_numeric_correlations"):
        lines.append("Top numeric correlations (abs):")
        for c in profile["top_numeric_correlations"][:5]:
            lines.append(f"  * {c['col_a']} <> {c['col_b']} : corr={c['corr']}")
    # quarterly trends
    if profile.get("quarterly_trends"):
        lines.append("Detected quarterly trends:")
        for k,v in profile["quarterly_trends"].items():
            lines.append(f"  * {k}: last={v['last_quarter_total']}, prev={v['prev_quarter_total']}, pct_change={v['pct_change']}%")
    text = "\n".join(lines)
    # limit lines to control token size
    text_lines = text.splitlines()[:max_lines]
    return "\n".join(text_lines)

def _fmt(x):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "NA"
    if isinstance(x, float):
        return f"{x:,.2f}"
    return str(x)

def summarize_dataframe_smart(df: pd.DataFrame, sheet_name: str="Sheet1") -> Tuple[Dict[str, Any], str]:
    """Return (structured_profile, text_summary) for a dataframe."""
    profile = dataframe_profile(df)
    # attach sheet name
    profile["_sheet_name"] = sheet_name
    text = f"Sheet: {sheet_name}\n" + profile_to_text(profile)
    return profile, text
