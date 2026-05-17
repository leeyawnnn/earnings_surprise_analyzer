"""
Statistical Analysis Module
Runs t-tests, correlations, and descriptive stats on earnings surprise data.
"""

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute descriptive statistics of returns by surprise category.
    
    Returns:
        DataFrame with stats for each category × return window.
    """
    categories = ["Beat", "In-Line", "Miss"]
    ret_cols = [c for c in df.columns if c.startswith("ret_day") and not c.endswith("_adj")]
    ret_adj_cols = [c for c in df.columns if c.endswith("_adj")]
    
    all_cols = ret_cols + ret_adj_cols
    
    records = []
    for cat in categories:
        subset = df[df["category"] == cat]
        for col in all_cols:
            vals = subset[col].dropna()
            if len(vals) == 0:
                continue
            records.append({
                "category": cat,
                "return_metric": col,
                "count": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
                "se": vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
            })
    
    stats_df = pd.DataFrame(records)
    
    if verbose:
        print("\n  ═══ Descriptive Statistics by Category ═══")
        for col in ret_cols:
            print(f"\n  {col}:")
            for cat in categories:
                row = stats_df[(stats_df["category"] == cat) & (stats_df["return_metric"] == col)]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(f"    {cat:8s}: mean={r['mean']:+.3f}%  median={r['median']:+.3f}%  "
                      f"std={r['std']:.3f}%  n={int(r['count'])}")
    
    return stats_df


def run_t_tests(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run two-sample t-tests comparing Beat vs Miss returns.
    Also tests Beat vs 0 and Miss vs 0 (one-sample).
    
    Returns:
        DataFrame with test results.
    """
    ret_cols = [c for c in df.columns if c.startswith("ret_day") and not c.endswith("_adj")]
    ret_adj_cols = [c for c in df.columns if c.endswith("_adj")]
    all_cols = ret_cols + ret_adj_cols
    
    beats = df[df["category"] == "Beat"]
    misses = df[df["category"] == "Miss"]
    
    results = []
    
    for col in all_cols:
        beat_vals = beats[col].dropna()
        miss_vals = misses[col].dropna()
        
        # ── Beat vs Miss (two-sample, Welch's t-test) ──────────────────
        if len(beat_vals) >= 2 and len(miss_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(beat_vals, miss_vals, equal_var=False)
            results.append({
                "test": f"Beat vs Miss",
                "return_metric": col,
                "beat_mean": beat_vals.mean(),
                "miss_mean": miss_vals.mean(),
                "diff": beat_vals.mean() - miss_vals.mean(),
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant_5pct": p_val < 0.05,
                "significant_10pct": p_val < 0.10,
                "n_beat": len(beat_vals),
                "n_miss": len(miss_vals),
            })
        
        # ── Beat returns vs 0 (one-sample) ─────────────────────────────
        if len(beat_vals) >= 2:
            t_stat, p_val = stats.ttest_1samp(beat_vals, 0)
            results.append({
                "test": "Beat vs 0",
                "return_metric": col,
                "beat_mean": beat_vals.mean(),
                "miss_mean": np.nan,
                "diff": beat_vals.mean(),
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant_5pct": p_val < 0.05,
                "significant_10pct": p_val < 0.10,
                "n_beat": len(beat_vals),
                "n_miss": np.nan,
            })
        
        # ── Miss returns vs 0 (one-sample) ─────────────────────────────
        if len(miss_vals) >= 2:
            t_stat, p_val = stats.ttest_1samp(miss_vals, 0)
            results.append({
                "test": "Miss vs 0",
                "return_metric": col,
                "beat_mean": np.nan,
                "miss_mean": miss_vals.mean(),
                "diff": miss_vals.mean(),
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant_5pct": p_val < 0.05,
                "significant_10pct": p_val < 0.10,
                "n_beat": np.nan,
                "n_miss": len(miss_vals),
            })
    
    test_df = pd.DataFrame(results)
    
    if verbose:
        print("\n  ═══ T-Test Results: Beat vs Miss ═══")
        bvm = test_df[test_df["test"] == "Beat vs Miss"]
        for _, row in bvm.iterrows():
            sig = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else ""))
            print(f"  {row['return_metric']:20s}: diff={row['diff']:+.3f}%  "
                  f"t={row['t_statistic']:.3f}  p={row['p_value']:.4f} {sig}")
    
    return test_df


def compute_correlations(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute Pearson correlation between surprise % and various return metrics.
    
    Returns:
        DataFrame with correlation results.
    """
    ret_cols = [c for c in df.columns if c.startswith("ret_day")]
    
    results = []
    for col in ret_cols:
        valid = df[["surprise_pct", col]].dropna()
        if len(valid) < 3:
            continue
        
        r, p = stats.pearsonr(valid["surprise_pct"], valid[col])
        
        # Also compute Spearman (more robust to outliers)
        rho, p_rho = stats.spearmanr(valid["surprise_pct"], valid[col])
        
        results.append({
            "return_metric": col,
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": p_rho,
            "n": len(valid),
        })
    
    corr_df = pd.DataFrame(results)
    
    if verbose:
        print("\n  ═══ Correlation: Surprise % vs Returns ═══")
        for _, row in corr_df.iterrows():
            sig = "***" if row["pearson_p"] < 0.01 else ("**" if row["pearson_p"] < 0.05 else ("*" if row["pearson_p"] < 0.10 else ""))
            print(f"  {row['return_metric']:20s}: r={row['pearson_r']:+.4f} (p={row['pearson_p']:.4f}) {sig}  "
                  f"ρ={row['spearman_rho']:+.4f} (p={row['spearman_p']:.4f})")
    
    return corr_df


def analyze_drift(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Analyze Post-Earnings Announcement Drift (PEAD).
    
    Checks whether the return gap between beats and misses widens over time
    (Day-0 → Day-5 → Day-20), suggesting momentum/drift.
    
    Returns:
        Dictionary with drift analysis results.
    """
    windows = [0, 1, 5, 10, 20]
    drift_results = {"raw": {}, "adjusted": {}}
    
    for suffix, label in [("", "raw"), ("_adj", "adjusted")]:
        for w in windows:
            col = f"ret_day{w}{suffix}"
            if col not in df.columns:
                continue
            
            beat_mean = df[df["category"] == "Beat"][col].dropna().mean()
            miss_mean = df[df["category"] == "Miss"][col].dropna().mean()
            
            if pd.isna(beat_mean) or pd.isna(miss_mean):
                continue
            
            drift_results[label][f"day_{w}"] = {
                "beat_mean": beat_mean,
                "miss_mean": miss_mean,
                "spread": beat_mean - miss_mean,
            }
    
    if verbose:
        print("\n  ═══ Post-Earnings Announcement Drift Analysis ═══")
        for label in ["raw", "adjusted"]:
            print(f"\n  {label.upper()} returns:")
            print(f"  {'Window':>10s} {'Beat':>10s} {'Miss':>10s} {'Spread':>10s}")
            print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
            for key, vals in drift_results[label].items():
                print(f"  {key:>10s} {vals['beat_mean']:>+9.3f}% {vals['miss_mean']:>+9.3f}% "
                      f"{vals['spread']:>+9.3f}%")
        
        # Check if spread increases over time (evidence of drift)
        raw_spreads = [(k, v["spread"]) for k, v in drift_results.get("raw", {}).items()]
        if len(raw_spreads) >= 2:
            if raw_spreads[-1][1] > raw_spreads[0][1]:
                print("\n  → Spread WIDENS over time → Evidence of PEAD ✓")
            else:
                print("\n  → Spread does NOT widen over time → Weak/no PEAD evidence")
    
    return drift_results


def run_full_analysis(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the complete statistical analysis suite."""
    print("\n" + "="*60)
    print("  STATISTICAL ANALYSIS")
    print("="*60)
    
    desc_stats = descriptive_stats(df, verbose=verbose)
    t_tests = run_t_tests(df, verbose=verbose)
    correlations = compute_correlations(df, verbose=verbose)
    drift = analyze_drift(df, verbose=verbose)
    
    return {
        "descriptive_stats": desc_stats,
        "t_tests": t_tests,
        "correlations": correlations,
        "drift": drift,
    }
