"""
utils_survival.py
=================
Shared utility functions for the Digital Twins Survival Analysis pipeline.

Contents:
    km_analysis                  — Kaplan-Meier summary statistics
    remove_highly_correlated_features — Pearson correlation filtering
    plot_km_curve                — Two-arm KM plot (solid lines)
    plot_km_curve_dashed         — Two-arm KM plot (customisable line styles)
    plot_multi_km                — Multi-arm KM plot with inline at-risk table
    compare_low_high             — Log-rank test + Cox HR for two groups
    safe_shap_summary_plot       — SHAP beeswarm summary (memory-safe)
    compare_cindex_from_ci       — Approximate test for ΔC-index
"""

##--------------------------------------------------------------------------------------------##
##                   km_analysis (Median PFS and N-month Survival Probability)                ##
##--------------------------------------------------------------------------------------------##
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def km_analysis(pre_pfs, real_pfs, drugC_data, drugB_data, time_point=60):
    """
    Compute Kaplan-Meier summary statistics for up to four survival datasets.

    Parameters
    ----------
    pre_pfs : structured np.ndarray
        Digital Twin survival data with fields ``PFS_time`` and ``PFS_events``.
    real_pfs : structured np.ndarray
        Observed survival data with fields ``PFS_time`` and ``PFS_events``.
    drugC_data : structured np.ndarray
        Survival data for drug arm C (real).
    drugB_data : structured np.ndarray
        Survival data for drug arm B (Digital Twin).
    time_point : float
        Landmark time at which to evaluate survival probability (same unit as
        ``PFS_time``, e.g. 60 = 60 months).

    Returns
    -------
    list of dict
        Each dict contains ``label``, ``median``, ``median_CI``,
        ``survival_at_t``, and ``survival_CI``.
    """

    def fit_and_summary(data, label):
        kmf = KaplanMeierFitter().fit(
            data["PFS_time"], event_observed=data["PFS_events"], label=label
        )
        median_surv = kmf.median_survival_time_
        ci_median   = median_survival_times(kmf.confidence_interval_)
        median_CI   = (ci_median.iloc[0, 0], ci_median.iloc[0, 1])

        surv_at_t = kmf.predict(time_point)
        ci_df = kmf.confidence_interval_
        idx   = np.argmin(np.abs(ci_df.index - time_point))
        surv_CI = (ci_df.iloc[idx, 0], ci_df.iloc[idx, 1])

        return dict(
            label=label,
            median=median_surv,
            median_CI=median_CI,
            survival_at_t=surv_at_t,
            survival_CI=surv_CI,
        )

    results = [
        fit_and_summary(pre_pfs,   "Digital Twin"),
        fit_and_summary(real_pfs,  "Observed"),
        fit_and_summary(drugC_data, "Drug Arm C — Real"),
        fit_and_summary(drugB_data, "Drug Arm B — DT"),
    ]

    for res in results:
        print(
            f"{res['label']}: Median = {res['median']:.2f} months "
            f"(95% CI = [{res['median_CI'][0]:.2f}, {res['median_CI'][1]:.2f}]), "
            f"{time_point}-month survival = {res['survival_at_t']:.2%} "
            f"(95% CI = {res['survival_CI'][0]:.2%} – {res['survival_CI'][1]:.2%})"
        )
    return results


##--------------------------------------------------------------------------------------------##
##                      Remove High-Correlation Features                                      ##
##--------------------------------------------------------------------------------------------##
import pandas as pd


def remove_highly_correlated_features(X, threshold=0.9, plot=False):
    """
    Remove features whose Pearson |r| exceeds *threshold* with any other feature.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    threshold : float
        Correlation threshold for removal (default 0.9).
    plot : bool
        If True, display a seaborn heatmap of the full correlation matrix.

    Returns
    -------
    X_filtered : pd.DataFrame
    removed_features : list of str
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_filtered = X.drop(columns=to_drop)

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True)
        plt.title(f"Correlation Matrix (threshold = {threshold})")
        plt.tight_layout()
        plt.show()

    return X_filtered, to_drop


##--------------------------------------------------------------------------------------------##
##                         KM Plotting Utilities                                              ##
##--------------------------------------------------------------------------------------------##
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def plot_km_curve(
    low_risk, high_risk, time_col, event_col, title,
    save_path=None, low_risk_color=None, high_risk_color=None,
    low_risk_label=None, high_risk_label=None,
):
    """
    Plot Kaplan-Meier curves for two risk groups (solid lines).

    Parameters
    ----------
    low_risk, high_risk : DataFrame or structured array
        Survival data for each group.
    time_col, event_col : str
        Column names for time and event indicator.
    title : str
        Plot title.
    save_path : str or None
        Path to save the PNG (300 dpi).
    low_risk_color, high_risk_color : str or None
        Matplotlib colour strings.  ``None`` uses the style default.
    low_risk_label, high_risk_label : str or None
        Legend labels.  Defaults to ``'Low Risk'`` / ``'High Risk'``.
    """
    plt.rcParams.update({"font.size": 16, "axes.labelweight": "bold"})
    fig, ax = plt.subplots(figsize=(9.0, 8.0))

    low_risk_label  = low_risk_label  or "Low Risk"
    high_risk_label = high_risk_label or "High Risk"

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(low_risk[time_col], event_observed=low_risk[event_col], label=low_risk_label)
    kmf_low.plot_survival_function(
        ax=ax, ci_show=True, linewidth=2.5, color=low_risk_color,
        show_censors=True, censor_styles={"marker": "|"},
    )

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(high_risk[time_col], event_observed=high_risk[event_col], label=high_risk_label)
    kmf_high.plot_survival_function(
        ax=ax, ci_show=True, linewidth=2.5, color=high_risk_color,
        show_censors=True, censor_styles={"marker": "|"},
    )

    for line in ax.lines:
        if line.get_linestyle() == "None" and line.get_marker() == "|":
            line.set_markersize(6)
            line.set_markeredgewidth(1.4)

    ax.xaxis.set_major_locator(MultipleLocator(12))
    add_at_risk_counts(kmf_low, kmf_high, ax=ax)

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Time (Months)", fontsize=16, fontweight="bold")
    plt.ylabel("Survival Probability", fontsize=16, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(18); lbl.set_fontweight("bold")

    handles, labels = ax.get_legend_handles_labels()
    lh = [h for h in handles if isinstance(h, Line2D)]
    ll = [labels[i] for i, h in enumerate(handles) if isinstance(h, Line2D)]
    leg = ax.legend(handles=lh, labels=ll, fontsize=16, loc="best", frameon=True)
    leg.get_frame().set_alpha(0); leg.get_frame().set_edgecolor("none")
    for txt in leg.get_texts():
        txt.set_fontsize(16)
    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


def plot_km_curve_dashed(
    low_risk, high_risk, time_col, event_col, title,
    save_path=None, low_risk_color=None, high_risk_color=None,
    low_risk_label=None, high_risk_label=None,
    low_risk_linestyle="-", high_risk_linestyle="--",
):
    """
    Plot Kaplan-Meier curves for two groups with configurable line styles.

    Parameters mirror :func:`plot_km_curve` with two additional arguments:

    low_risk_linestyle : str
        Matplotlib line style for the low-risk group (default ``'-'``).
    high_risk_linestyle : str
        Matplotlib line style for the high-risk group (default ``'--'``).
    """
    plt.rcParams.update({"font.size": 16, "axes.labelweight": "bold"})
    fig, ax = plt.subplots(figsize=(9.0, 8.0))

    low_risk_label  = low_risk_label  or "Low Risk"
    high_risk_label = high_risk_label or "High Risk"

    for data, label, color, ls in [
        (low_risk,  low_risk_label,  low_risk_color,  low_risk_linestyle),
        (high_risk, high_risk_label, high_risk_color, high_risk_linestyle),
    ]:
        kmf = KaplanMeierFitter()
        kmf.fit(data[time_col], event_observed=data[event_col], label=label)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, linewidth=2.5,
            color=color, linestyle=ls,
            show_censors=True, censor_styles={"marker": "|"},
        )

    ax.xaxis.set_major_locator(MultipleLocator(12))
    add_at_risk_counts(*[
        KaplanMeierFitter().fit(d[time_col], event_observed=d[event_col], label=l)
        for d, l in [(low_risk, low_risk_label), (high_risk, high_risk_label)]
    ], ax=ax)

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Time (Months)", fontsize=16, fontweight="bold")
    plt.ylabel("Survival Probability", fontsize=16, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(18); lbl.set_fontweight("bold")

    handles, labels = ax.get_legend_handles_labels()
    lh = [h for h in handles if isinstance(h, Line2D)]
    ll = [labels[i] for i, h in enumerate(handles) if isinstance(h, Line2D)]
    leg = ax.legend(handles=lh, labels=ll, fontsize=16, loc="best", frameon=True)
    leg.get_frame().set_alpha(0); leg.get_frame().set_edgecolor("none")
    for txt in leg.get_texts():
        txt.set_fontsize(16)
    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


def _draw_inline_at_risk(ax, kmf_dict, xticks):
    """Draw per-subgroup at-risk counts as inline text below the KM axes."""
    y0, dy = -0.23, 0.07
    xlims = ax.get_xlim()

    def x_to_axes(x):
        if x <= xlims[0]: return 0
        if x >= xlims[1]: return 1
        return (x - xlims[0]) / (xlims[1] - xlims[0])

    xticks = xticks[:-1]
    for r, (label, kmf) in enumerate(kmf_dict.items()):
        row_y = y0 - r * dy
        ax.text(-0.02, row_y, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=16, fontweight="bold")
        ev_tab = kmf.event_table
        for x in xticks:
            pos      = ev_tab.index.get_indexer([x], method="ffill")[0]
            at_risk  = int(ev_tab["at_risk"].iloc[0] if pos == -1 else ev_tab["at_risk"].iloc[pos])
            ax.text(x_to_axes(x), row_y, str(at_risk), transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")

    ax.axhline(y=ax.get_ylim()[0], xmin=0, xmax=1, color="black", linewidth=0.5)


def plot_multi_km(
    subgroups, time_col, event_col, title,
    styles=None, ci_show=True, save_path=None, legend_path=None,
):
    """
    Plot KM curves for multiple subgroups with an inline at-risk table.

    Parameters
    ----------
    subgroups : dict {label: DataFrame}
        Mapping from group label to survival DataFrame.
    time_col, event_col : str
        Column names for survival time and event indicator.
    title : str
        Plot title.
    styles : dict or None
        Optional per-label style dict with keys ``color``, ``ls``, ``lw``.
    ci_show : bool
        Whether to display confidence intervals.
    save_path : str or None
        Path to save the main KM figure.
    legend_path : str or None
        Path to save a standalone legend figure.
    """
    plt.rcParams.update({"font.size": 18, "axes.labelweight": "bold"})
    fig, ax = plt.subplots(figsize=(10, 6.8))

    if styles is None:
        styles = {
            "Low Risk (Drug A)":  {"color": "#467821", "ls": "-",  "lw": 2.5},
            "Low Risk (Drug B)":  {"color": "#0072B2", "ls": "-",  "lw": 2.5},
            "High Risk (Drug A)": {"color": "#D55E00", "ls": "-",  "lw": 2.5},
            "High Risk (Drug B)": {"color": "#A60628", "ls": "-",  "lw": 2.5},
        }

    kmf_dict = {}
    for label, df in subgroups.items():
        if len(df) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], event_observed=df[event_col], label=label)
        s = styles.get(label, {})
        censor_styles = {"marker": "|", "markeredgecolor": s.get("color", "k")}
        kmf.plot_survival_function(
            ax=ax, ci_show=ci_show,
            show_censors=True, censor_styles=censor_styles,
            color=s.get("color"), linestyle=s.get("ls", "-"),
            linewidth=s.get("lw", 2.5),
        )
        kmf_dict[label] = kmf

    for line in ax.lines:
        if line.get_linestyle() == "None" and line.get_marker() == "|":
            line.set_markersize(6); line.set_markeredgewidth(1.8)

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Time (Months)", fontsize=18, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(18); lbl.set_fontweight("bold")

    handles, labels = ax.get_legend_handles_labels()
    line_pairs = [(h, l) for h, l in zip(handles, labels) if isinstance(h, Line2D)]
    line_handles, line_labels = zip(*line_pairs) if line_pairs else ([], [])
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    if legend_path:
        lfig = plt.figure(figsize=(4, max(1, 0.6 * len(line_labels))))
        lax = lfig.add_subplot(111); lax.axis("off")
        if line_handles:
            lax.legend(list(line_handles), list(line_labels),
                       fontsize=14, loc="center", frameon=False)
        lfig.savefig(legend_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(lfig)

    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_color("black")

    ax.xaxis.set_major_locator(MultipleLocator(12))
    xticks = np.array([t for t in ax.get_xticks() if t >= 0])
    if len(xticks) > 0:
        _draw_inline_at_risk(ax, kmf_dict, xticks)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    plt.close()


##--------------------------------------------------------------------------------------------##
##                      HR and log-rank comparison helpers                                    ##
##--------------------------------------------------------------------------------------------##
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


def subgroup_to_df(arr, time_col="PFS_time", event_col="PFS_events"):
    """Convert a structured array subgroup into a plain DataFrame."""
    return pd.DataFrame({
        time_col:  arr[time_col],
        event_col: arr[event_col].astype(int),
    })


def _truncate_at_time(df, tmax, time_col, event_col):
    """Administratively censor observations beyond *tmax*."""
    out = df[[time_col, event_col]].copy()
    over = out[time_col] > tmax
    out.loc[over, event_col] = 0
    out.loc[over, time_col]  = tmax
    return out


def compare_low_high(
    low_arr, high_arr,
    time_col="PFS_time", event_col="PFS_events",
    logrank_tmax=None, truncate_hr=False,
):
    """
    Compute Cox HR (High vs Low) and log-rank p-value.

    Returns
    -------
    tuple
        ``(hr, ci_low, ci_up, pval, c_index,
           hr_ref_high, ci_low_ref_high, ci_up_ref_high)``
    """
    low  = subgroup_to_df(low_arr,  time_col, event_col)
    high = subgroup_to_df(high_arr, time_col, event_col)

    low_hr, high_hr = low, high
    if truncate_hr and logrank_tmax is not None:
        low_hr  = _truncate_at_time(low,  logrank_tmax, time_col, event_col)
        high_hr = _truncate_at_time(high, logrank_tmax, time_col, event_col)

    low_hr  = low_hr.copy();  low_hr["Risk"]  = 0
    high_hr = high_hr.copy(); high_hr["Risk"] = 1
    df_hr   = pd.concat([low_hr, high_hr], ignore_index=True)

    cph = CoxPHFitter()
    cph.fit(df_hr[[time_col, event_col, "Risk"]], duration_col=time_col, event_col=event_col)
    s = cph.summary.loc["Risk"]
    hr, ci_low, ci_up = float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])
    c_index = float(cph.concordance_index_)

    hr_ref_high    = 1.0 / hr
    ci_low_rh      = 1.0 / ci_up
    ci_up_rh       = 1.0 / ci_low

    low_lr, high_lr = (
        (_truncate_at_time(low,  logrank_tmax, time_col, event_col),
         _truncate_at_time(high, logrank_tmax, time_col, event_col))
        if logrank_tmax is not None else (low, high)
    )
    lr = logrank_test(
        low_lr[time_col], high_lr[time_col],
        event_observed_A=low_lr[event_col],
        event_observed_B=high_lr[event_col],
    )
    pval = float(lr.p_value)

    return hr, ci_low, ci_up, pval, c_index, hr_ref_high, ci_low_rh, ci_up_rh


##--------------------------------------------------------------------------------------------##
##                      SHAP summary plot (memory-safe)                                       ##
##--------------------------------------------------------------------------------------------##
import os
import shap


def safe_shap_summary_plot(
    model, x_test_df, feature_names, save_dir,
    fold_name="Fold", max_samples=500, top_k=20,
):
    """
    Compute SHAP values with TreeExplainer and save a beeswarm summary plot.

    Parameters
    ----------
    model : xgboost.Booster
        Trained model (loaded on CPU).
    x_test_df : pd.DataFrame
        Test feature matrix.
    feature_names : list of str
    save_dir : str
        Directory for output files.
    fold_name : str
        Label used in file names and plot title.
    max_samples : int
        Maximum number of samples used for SHAP (for memory efficiency).
    top_k : int
        Number of top features to display and save.
    """
    print(f"[{fold_name}] Computing SHAP values …")
    os.makedirs(save_dir, exist_ok=True)

    explainer   = shap.TreeExplainer(model)
    x_np        = x_test_df.to_numpy()
    shap_values = explainer.shap_values(x_np)

    N = min(max_samples, x_np.shape[0])
    sv_sub = shap_values[:N]
    x_sub  = x_test_df.iloc[:N]

    mean_abs  = np.abs(shap_values).mean(axis=0)
    topk_idx  = np.argsort(mean_abs)[-top_k:][::-1]
    sv_top    = sv_sub[:, topk_idx]
    x_top     = x_sub.iloc[:, topk_idx]
    top_names = [feature_names[i] for i in topk_idx]

    shap.summary_plot(sv_top, x_top, feature_names=top_names, plot_type="dot", show=False)
    plt.title(f"SHAP Summary — {fold_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"SHAP_summary_{fold_name}.png"), dpi=300)
    plt.close()

    pd.DataFrame({"Feature": top_names, "MeanAbsSHAP": mean_abs[topk_idx]}).to_csv(
        os.path.join(save_dir, f"SHAP_importance_{fold_name}.csv"), index=False
    )
    print(f"[{fold_name}] SHAP outputs saved to {save_dir}")


##--------------------------------------------------------------------------------------------##
##                      C-index comparison (approximate, CI-based)                           ##
##--------------------------------------------------------------------------------------------##
import math


def _se_from_ci(c, lo, hi):
    halfwidth = max(hi - c, c - lo)
    return halfwidth / 1.96


def compare_cindex_from_ci(c1, lo1, hi1, c2, lo2, hi2,
                           paired=True, rho=0.8, alternative="two-sided"):
    """
    Approximate test for ΔC = C2 − C1 using only reported confidence intervals.

    Parameters
    ----------
    c1, lo1, hi1 : float
        C-index and 95% CI limits for model 1.
    c2, lo2, hi2 : float
        C-index and 95% CI limits for model 2.
    paired : bool
        True if both C-indices were computed on the same patients.
    rho : float
        Assumed Pearson correlation between the two C-indices (paired only).
    alternative : str
        ``'two-sided'``, ``'greater'`` (C2 > C1), or ``'less'`` (C2 < C1).

    Returns
    -------
    dict with keys ``delta``, ``SE_delta``, ``CI_low``, ``CI_high``, ``z``, ``p``.
    """
    se1, se2 = _se_from_ci(c1, lo1, hi1), _se_from_ci(c2, lo2, hi2)
    var_delta = se1**2 + se2**2 - (2.0 * rho * se1 * se2 if paired else 0)

    if not math.isfinite(var_delta) or var_delta <= 0:
        return {k: float("nan") for k in ["delta", "SE_delta", "CI_low", "CI_high", "z", "p"]}

    se_d  = var_delta ** 0.5
    delta = c2 - c1
    z     = delta / se_d

    def _phi(z_):
        return 0.5 * (1.0 + math.erf(z_ / math.sqrt(2.0)))

    alt = alternative.lower()
    p = (2.0 * min(_phi(z), 1.0 - _phi(z)) if alt == "two-sided"
         else (1.0 - _phi(z) if alt == "greater" else _phi(z)))

    return {
        "delta": delta, "SE_delta": se_d,
        "CI_low": delta - 1.96 * se_d, "CI_high": delta + 1.96 * se_d,
        "z": z, "p": p,
    }
