"""
=============================================================================
Calibration & Individual Treatment Effect (ITE) Utilities
=============================================================================
Functions for landmark-corrected, covariate-adjusted ITE estimation of a
post-baseline treatment (e.g. Local Consolidative Therapy / LCT) and for
plotting per-patient counterfactual survival curves.

Provided in this module
------------------------
    density_ratio_logit                    — 1-D transport weight estimator
    density_ratio_logit_multifeature       — multivariate transport weight estimator
    estimate_LCT_ITE_for_test              — main ITE estimation function
    plot_lct_counterfactual_for_patient    — per-patient counterfactual plot

NOT provided (supply your own implementation)
------------------------------------------------
    full_calibration_pipeline              — drug-stratified calibrator fitting
    plot_ibs_curve_DT                      — integrated Brier score curve
    plot_patients_survival_DT_split_legend — multi-patient DT curve plot
    WeightedParametricCalibrator           — parametric survival calibrator
                                              class with the following interface:
        .fit(r, T, E, weights=None)                 -> self
        .predict_survival_at_times(r, months_grid)   -> np.ndarray (n, t)
        .predict_median(r)                           -> np.ndarray (n,)
        .predict_expectation(r)                      -> np.ndarray (n,)

Contents:
    full_calibration_pipeline                       — drug-stratified calibrator fitting
    plot_ibs_curve_DT                               — integrated Brier score calculation
    plot_patients_survival_DT_split_legend          — individual digital-twin based survival prediction
    estimate_LCT_ITE_for_test                       — Individual treatment effect estimation
    plot_lct_counterfactual_for_patient             — Pearson correlation filtering
""" 

"""
Dependencies
------------
numpy, pandas, scikit-learn, matplotlib
=============================================================================
"""


# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter, WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter

# ===================== Config (you can override in run_full_pipeline) =====================
DEFAULT_EVAL_TIMES = [6, 12, 24]
DEFAULT_N_BINS     = 4
DEFAULT_TAU_IBS    = 60

# =========================== General helpers =====================
def km_censoring(times, events):
    times = np.asarray(times, float); events = np.asarray(events, int)
    km = KaplanMeierFitter()
    km.fit(durations=times, event_observed=(1 - events), label="G_hat")
    return km

def brier_ipcw(times, events, p_t, t, kmc=None, clip=1e-10):
    times = np.asarray(times, float); events = np.asarray(events, int); p_t = np.asarray(p_t, float)
    if kmc is None: kmc = km_censoring(times, events)
    Gt  = np.clip(kmc.predict(t), clip, 1.0)
    eps = 1e-12
    GTm = np.clip(kmc.predict(np.maximum(times - eps, 0.0)), clip, 1.0)
    Y   = (times > t).astype(float)
    w   = np.zeros_like(times, float)
    w[times > t] = 1.0 / Gt
    m2 = (times <= t) & (events == 1)
    w[m2] = 1.0 / GTm[m2]
    return float(np.mean(w * (Y - p_t) ** 2))

def ibs_ipcw(times, events, S_mat, grid_times, kmc=None):
    if kmc is None: kmc = km_censoring(times, events)
    bs_vals = np.array([brier_ipcw(times, events, S_mat[:, j], t, kmc) for j, t in enumerate(grid_times)])
    tau = grid_times[-1]
    return float(np.trapz(bs_vals, grid_times) / tau), bs_vals

def calib_bins_at_t(times, events, p_t, t, n_bins=10):
    times = np.asarray(times, float); events = np.asarray(events, int); p_t = np.asarray(p_t, float)
    q = np.linspace(0, 1, n_bins + 1); edges = np.quantile(p_t, q)
    edges[0] -= 1e-12; edges[-1] += 1e-12
    bid = np.digitize(p_t, edges) - 1; bid = np.clip(bid, 0, n_bins - 1)
    km = KaplanMeierFitter(); rows = []; obs_assign = np.zeros_like(p_t, float)
    for b in range(n_bins):
        idx = (bid == b)
        if not np.any(idx):
            rows.append(dict(bin=b, p=np.nan, obs=np.nan, count=0)); continue
        km.fit(durations=times[idx], event_observed=events[idx])
        s_t = float(km.predict(t)); p_mean = float(p_t[idx].mean())
        rows.append(dict(bin=b, p=p_mean, obs=s_t, count=int(idx.sum())))
        obs_assign[idx] = s_t
    df = pd.DataFrame(rows)
    abs_err = np.abs(obs_assign - p_t)
    return df, abs_err

def ici_from_abs_err(abs_err):
    arr = np.asarray(abs_err, float)
    return float(np.nanmean(arr))

# ===================== Calibrator & selection helpers =====================
class ParametricCalibrator:
    def __init__(self, family_name="weibull"):
        self.family_name = family_name
        self.model = {"weibull": WeibullAFTFitter(),
                      "loglogistic": LogLogisticAFTFitter(),
                      "lognormal": LogNormalAFTFitter()}[family_name]
        self.fitted = False

    def fit(self, r, T, E):
        df = pd.DataFrame({"T": np.asarray(T, float),
                           "E": np.asarray(E, int),
                           "r": np.asarray(r, float)})
        self.model.fit(df, duration_col="T", event_col="E")
        self.fitted = True
        return self

    def predict_surv(self, r, times_grid):
        sf = self.model.predict_survival_function(pd.DataFrame({"r": np.asarray(r, float)}),
                                                  times=times_grid)
        return sf.values.T  # (n, len(times))

    # lifelines-style convenience
    def predict_survival_at_times(self, r, times_grid):
        return self.predict_surv(r, times_grid)

    def predict_median(self, r):
        df = pd.DataFrame({"r": np.asarray(r, float)})
        return np.asarray(self.model.predict_median(df)).astype(float)

    def predict_expectation(self, r):
        df = pd.DataFrame({"r": np.asarray(r, float)})
        return np.asarray(self.model.predict_expectation(df)).astype(float)

def select_best_family_by_ibs(r, T, E, months_grid, TAU_IBS=60, val_ratio=0.25, seed=42):
    idx = np.arange(len(T))
    if len(idx) < 12:
        return "weibull", np.nan
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed)
    families = ["weibull", "loglogistic", "lognormal"]
    kmc = km_censoring(T[va], E[va])
    mask_tau = months_grid <= min(TAU_IBS, months_grid.max())
    best_name, best_ibs = None, 1e9
    for fam in families:
        cal = ParametricCalibrator(fam).fit(r[tr], T[tr], E[tr])
        S_val = cal.predict_surv(r[va], months_grid)
        ibs, _ = ibs_ipcw(T[va], E[va], S_val[:, mask_tau], months_grid[mask_tau], kmc)
        if ibs < best_ibs:
            best_ibs, best_name = ibs, fam
    return best_name, best_ibs

def fit_calibrators_by_group_custom(groups_tr, r_tr, T_tr, E_tr, months_grid, manual_choice=None, TAU_IBS=60):
    manual_choice = manual_choice or {}
    calibrators, chosen = {}, {}
    for g in np.unique(groups_tr):
        idx = (groups_tr == g)
        fam = manual_choice.get(g, None)
        if fam is None:
            fam, ibs_est = select_best_family_by_ibs(r_tr[idx], T_tr[idx], E_tr[idx], months_grid, TAU_IBS=TAU_IBS)
        else:
            ibs_est = np.nan
        cal = ParametricCalibrator(fam).fit(r_tr[idx], T_tr[idx], E_tr[idx])
        calibrators[g] = cal
        chosen[g] = (fam, ibs_est)
        msg_ibs = "" if np.isnan(ibs_est) else f", IBS≈{ibs_est:.3f}"
        print(f"[Calibrator] Drug={g}: {fam}{msg_ibs}")

    # Calculate the IBS for overall cohort (if error, can remove this part)
    uniq, counts = np.unique(groups_tr, return_counts=True)
    group_sizes = {g: int(c) for g, c in zip(uniq, counts)}
    group_ibs   = {g: chosen.get(g, (None, np.nan))[1] for g in uniq}
    valid = [g for g in uniq if np.isfinite(group_ibs[g])]
    w = np.array([group_sizes[g] for g in valid], float)
    v = np.array([group_ibs[g]   for g in valid], float)
    overall_ibs_weighted = float(np.sum(w * v) / np.sum(w))
    print(f"[Calibrator] All: IBS≈{overall_ibs_weighted:.3f}")
    
    return calibrators, chosen

# ===================== Group prediction helpers =====================
def predict_S_by_group(calibrators, groups, r, months_grid):
    S = np.zeros((len(r), len(months_grid)))
    for g, cal in calibrators.items():
        idx = (groups == g)
        if np.any(idx):
            S[idx, :] = cal.predict_survival_at_times(r[idx], months_grid)
    return S

def predict_median_by_group(calibrators, groups, r):
    out = np.full(len(r), np.nan, dtype=float)
    for g, cal in calibrators.items():
        idx = (groups == g)
        if np.any(idx):
            out[idx] = cal.predict_median(r[idx])
    return out

def predict_expectation_by_group(calibrators, groups, r):
    out = np.full(len(r), np.nan, dtype=float)
    for g, cal in calibrators.items():
        idx = (groups == g)
        if np.any(idx):
            out[idx] = cal.predict_expectation(r[idx])
    return out

# ===================== Pseudo PFS simulation helpers =====================
def _invert_survival_to_time(u, t_grid, s_row):
    t_grid = np.asarray(t_grid, float); s_row = np.asarray(s_row, float)
    s_rev = s_row[::-1]; t_rev = t_grid[::-1]
    s_rev_nd = np.maximum.accumulate(s_rev)
    xp, idx = np.unique(s_rev_nd, return_index=True)
    fp = t_rev[idx]
    u_clipped = np.clip(u, xp[0], xp[-1])
    return float(np.interp(u_clipped, xp, fp))

def _sample_event_times_for_group(calibrator, r_vec, tau_max, step=0.1, seed=42):
    rng = np.random.RandomState(seed)
    r_vec = np.asarray(r_vec, float)
    t_dense = np.arange(0.0, float(tau_max) + step, step, dtype=float)
    S_dense = calibrator.predict_survival_at_times(r_vec, t_dense)  # (n, len(t_dense))
    u = rng.rand(len(r_vec))
    return np.array([_invert_survival_to_time(ui, t_dense, S_dense[i]) for i, ui in enumerate(u)], float)

def _sample_censor_times_from_km(km_censor, n, tau_max, seed=123):
    rng = np.random.RandomState(seed)
    sf = km_censor.survival_function_
    t_km = sf.index.values.astype(float)
    g    = sf.iloc[:, 0].values.astype(float)
    if t_km[0] > 0:
        t_km = np.r_[0.0, t_km]; g = np.r_[1.0, g]
    if t_km[-1] < tau_max:
        t_km = np.r_[t_km, tau_max]; g = np.r_[g, float(km_censor.predict(tau_max))]
    u = rng.rand(n)
    g_rev = g[::-1]; t_rev = t_km[::-1]
    g_rev_nd = np.maximum.accumulate(g_rev)
    xp, idx = np.unique(g_rev_nd, return_index=True)
    fp = t_rev[idx]
    u_clip = np.clip(u, xp[0], xp[-1])
    return np.interp(u_clip, xp, fp).astype(float)

def simulate_pfs_pairs(calibrators, groups, r, tau_max, km_censor=None,
                       step=0.1, seed_event=42, seed_censor=123, admin_censor=None):
    groups = np.asarray(groups); r = np.asarray(r, float)
    n = len(r); T_event = np.full(n, np.nan, float)
    for g, cal in calibrators.items():
        idx = (groups == g)
        if not np.any(idx): continue
        T_event[idx] = _sample_event_times_for_group(cal, r[idx], tau_max, step=step, seed=seed_event)
    if admin_censor is not None:
        C = np.full(n, float(admin_censor), float)
    elif km_censor is not None:
        C = _sample_censor_times_from_km(km_censor, n, tau_max, seed=seed_censor)
    else:
        C = np.full(n, float(tau_max), float)
    T_obs = np.minimum(T_event, C)
    E_obs = (T_event <= C).astype(int)
    return T_obs, E_obs

def assign_median_as_event_time(calibrators, groups, r, admin_censor=None, tau_max=None):
    groups = np.asarray(groups); r = np.asarray(r, float)
    n = len(r)
    med = np.full(n, np.nan, float)
    for g, cal in calibrators.items():
        idx = (groups == g)
        if np.any(idx): med[idx] = cal.predict_median(r[idx])
    horizon = admin_censor if admin_censor is not None else (tau_max if tau_max is not None else np.nanmax(med))
    T_obs = np.minimum(med, horizon)
    E_obs = (med <= horizon).astype(int)
    return T_obs, E_obs

def pretty_drug_name(g):
    s = str(g).strip().lower()
    if s in {"1", "drug-1", "drug1"}: return "1ˢᵗ gen TKI"   # Drug-1
    if s in {"2", "drug-2", "drug2"}: return "2ⁿᵈ gen TKI"   # Drug-2
    return str(g)

# ===================== Plotting calibration curves helpers =====================
def plot_calibration_curves_test(
    T_test, E_test, DRUG_test, S_test, months_grid, EVAL_TIMES,
    N_BINS=4, 
    outpath=None,              # e.g., "./calibration_drug_by_time_TEST.png"
    figsize=(6, 5.4), dpi=300,
    LINE_WIDTH=2, MARKER_SIZE=9, FRAME_WIDTH=1.3,
    LABEL_SIZE=16, TICK_SIZE=16, LEGEND_SIZE=12,
    show=True
):
    from matplotlib.font_manager import FontProperties

    # Your latest mapping: Drug-1 -> blue; Drug-2 -> orange
    def color_for_drug(g):
        s = str(g).strip().lower()
        if s in {"1", "drug-1", "drug1"}: return "blue"
        if s in {"2", "drug-2", "drug2"}: return "orange"
        return "gray"

    marker_cycle = ['o', 's', '^', 'D', 'P', 'X']
    time_markers = {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(EVAL_TIMES)}

    def fmt_keep_y_hide_x_x(x, pos): 
        return "" if np.isclose(x, 0.0) else f"{x:.1f}"
    def fmt_keep_y_hide_x_y(y, pos): 
        return "0" if np.isclose(y, 0.0) else f"{y:.1f}"

    kmc_te = km_censoring(T_test, E_test)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot([0, 1], [0, 1], "--", lw=2, color="black", label="Ideal")

    metrics_rows_test = []
    drug_values = list(np.unique(DRUG_test))

    for t in EVAL_TIMES:
        j = int(np.where(months_grid == t)[0][0])
        for g in drug_values:
            idx = (np.asarray(DRUG_test) == g)
            if not np.any(idx): 
                continue

            p_t = S_test[idx, j]
            df_bins, abs_err = calib_bins_at_t(
                np.asarray(T_test)[idx], np.asarray(E_test)[idx],
                p_t, t, n_bins=N_BINS
            )
            df_plot = df_bins.dropna(subset=["p", "obs"]).sort_values("p")

            label_text = f"{pretty_drug_name(g)} ({t}mo)"
            ax.plot(
                df_plot["p"].values, df_plot["obs"].values,
                linestyle="-", color=color_for_drug(g),
                marker=time_markers[t], ms=MARKER_SIZE, lw=LINE_WIDTH,
                mec="white", mew=0.5, label=label_text
            )

            ici   = ici_from_abs_err(abs_err)
            brier = brier_ipcw(
                np.asarray(T_test)[idx], np.asarray(E_test)[idx],
                p_t, t, kmc=kmc_te
            )
            metrics_rows_test.append(
                dict(Drug=str(g), Time=t, ICI=ici, Brier=brier, n=int(np.sum(idx)))
            )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for side in ("left", "bottom", "right", "top"):
        ax.spines[side].set_linewidth(FRAME_WIDTH); ax.spines[side].set_color("black")
    ax.set_xlabel("Predicted Survival Probability", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Observed Survival Probability", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=5, width=1.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(TICK_SIZE); tick.set_fontweight("bold")

    from matplotlib.font_manager import FontProperties
    legend_font = FontProperties(size=LEGEND_SIZE, weight="bold")
    ax.legend(prop=legend_font, ncol=1, markerscale=1.1, handlelength=2.0, frameon=False, loc='upper left') ##
    ax.grid(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_keep_y_hide_x_x))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_keep_y_hide_x_y))

    plt.tight_layout()
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)

    if outpath is not None:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.05, transparent=True)
    if show:
        plt.show()

    df_metrics = (
        pd.DataFrame(metrics_rows_test)
          .sort_values(["Drug","Time"])
          .set_index(["Drug","Time"])
    )
    return fig, ax, df_metrics

# ===================== IBS by drug helpers =====================
def compute_ibs_by_drug_test(T_test, E_test, DRUG_test, S_test, months_grid, TAU_IBS=60):
    kmc_te = km_censoring(T_test, E_test)
    mask_tau = (months_grid <= TAU_IBS)
    ibs_rows = []
    for g in np.unique(DRUG_test):
        idx = (np.asarray(DRUG_test) == g)
        if np.sum(idx) == 0: 
            continue
        ibs_g, _ = ibs_ipcw(np.asarray(T_test)[idx], np.asarray(E_test)[idx],
                            S_test[idx][:, mask_tau], months_grid[mask_tau], kmc=kmc_te)
        ibs_rows.append(dict(Drug=str(g), IBS=f"{ibs_g:.4f}", n=int(np.sum(idx))))
    ibs_all, _ = ibs_ipcw(np.asarray(T_test), np.asarray(E_test),
                          S_test[:, mask_tau], months_grid[mask_tau], kmc=kmc_te)
    ibs_rows.append(dict(Drug="ALL", IBS=f"{ibs_all:.4f}", n=len(T_test)))
    return pd.DataFrame(ibs_rows).set_index("Drug")



# ===================== 1) Fits drug-stratified survival calibrators =====================
def full_calibration_pipeline(
    T_train, E_train, DRUG_train, T_test, E_test, DRUG_test,
    r_train_oof, r_test,
    EVAL_TIMES=DEFAULT_EVAL_TIMES, N_BINS=DEFAULT_N_BINS, TAU_IBS=DEFAULT_TAU_IBS,
    CALIBRATOR_CHOICE=DEFAULT_CALIBRATOR_CHOICE,
    outdir=".", save_plot=True, show_plot=True,
    style_kwargs=None
):
    """
    Full run:
    - Fit per-drug parametric calibrators on TRAIN.
    - Predict S_test on a months grid (covering eval times + TAU_IBS).
    - Plot TEST-set calibration curves (color by drug, marker by time) as in your style.
    - Compute metrics (ICI/Brier @ times; IBS @ [0,TAU_IBS]).
    - Produce pseudo PFS times by stochastic simulation + deterministic (median).
    Returns a dict of results.
    """
    # 1) time grid
    months_grid = np.arange(1, max(max(EVAL_TIMES), TAU_IBS) + 1)

    # 2) fit calibrators
    calibrators, chosen = fit_calibrators_by_group_custom(
        np.asarray(DRUG_train), np.asarray(r_train_oof, float),
        np.asarray(T_train, float), np.asarray(E_train, int),
        months_grid=months_grid, manual_choice=CALIBRATOR_CHOICE, TAU_IBS=TAU_IBS
    )

    # 3) predict survival
    S_test  = predict_S_by_group(calibrators, np.asarray(DRUG_test),  np.asarray(r_test, float),      months_grid)

    # ensure eval times exist on grid
    for t in EVAL_TIMES:
        if t not in set(months_grid.tolist()):
            months_grid = np.sort(np.unique(np.append(months_grid, t)))
            S_test  = predict_S_by_group(calibrators, np.asarray(DRUG_test),  np.asarray(r_test, float), months_grid)

    # 3b) individual medians & expectations for TEST (used in pseudo df)
    median_test   = predict_median_by_group(calibrators, np.asarray(DRUG_test),  np.asarray(r_test, float))
    expected_test = predict_expectation_by_group(calibrators, np.asarray(DRUG_test),  np.asarray(r_test, float))

    # 4) plot TEST-set calibration
    outpath = f"{outdir}/calibration_by_drug_time.png" if save_plot else None
    style_defaults = dict(figsize=(6,5.4), dpi=300, LINE_WIDTH=2, MARKER_SIZE=9,
                          FRAME_WIDTH=1.3, LABEL_SIZE=16, TICK_SIZE=16, LEGEND_SIZE=12)
    if style_kwargs:
        style_defaults.update(style_kwargs)
    fig, ax, df_metrics_test = plot_calibration_curves_test(
        T_test, E_test, DRUG_test, S_test, months_grid, EVAL_TIMES,
        N_BINS=N_BINS, outpath=outpath, show=show_plot, **style_defaults
    )

    # 5) IBS table (TEST)
    df_ibs_test = compute_ibs_by_drug_test(T_test, E_test, DRUG_test, S_test, months_grid, TAU_IBS=TAU_IBS)
    print("\n[IBS @ 0–{} mo] (TEST)".format(TAU_IBS))
    print(df_ibs_test)

    # 6) pseudo PFS times (TEST)
    tau_max = float(months_grid[-1])  # or max(T_test)
    kmc_te  = km_censoring(T_test, E_test)

    PFS_time_sim_test, PFS_event_sim_test = simulate_pfs_pairs(
        calibrators=calibrators,
        groups=np.asarray(DRUG_test),
        r=np.asarray(r_test, float),
        tau_max=tau_max,
        km_censor=kmc_te,
        step=0.1, seed_event=2024, seed_censor=2025,
        admin_censor=None
    )

    # # Ootional (with robust estimation)
    # T_obs, E_obs = simulate_pfs_pairs_Robust(
    #     calibrators=calibrators,
    #     groups=groups,
    #     r=r,
    #     tau_max=60.0,
    #     km_censor=None,          # no KM-based censoring
    #     admin_censor=60.0,       # administrative cutoff (months)
    #     step=0.1,               # grid resolution (months)
    #     seed_event=42,
    #     seed_censor=123
    # )

    df_pseudo_test = pd.DataFrame({
        "Drug": np.asarray(DRUG_test),
        "risk_r": np.asarray(r_test, float),
        "PFS_time_sim": PFS_time_sim_test,
        "PFS_event_sim": PFS_event_sim_test,
        "median_pred": median_test,
        "mean_pred": expected_test
    })
    print("\n[Test] pseudo-observed (simulated) PFS head:")
    print(df_pseudo_test.head())

    # Deterministic (median-based)
    PFS_time_med_test, PFS_event_med_test = assign_median_as_event_time(
        calibrators=calibrators,
        groups=np.asarray(DRUG_test),
        r=np.asarray(r_test, float),
        admin_censor=tau_max
    )

    results = dict(
        months_grid=months_grid,
        calibrators=calibrators,
        chosen=chosen,
        Surv=S_test,
        median_surv=median_test,
        expected_surv=expected_test,
        df_metrics_all=df_metrics_test,
        df_ibs=df_ibs_test,
        df_pseudo_surv=df_pseudo_test,
        PFS_time_sim=PFS_time_sim_test,
        PFS_event_sim=PFS_event_sim_test,
        PFS_time_med=PFS_time_med_test,
        PFS_event_med=PFS_event_med_test,
        fig_calib=fig,
        ax_calib=ax
    )
    return results


# ===================== 2) Plots time-varying Brier score =====================
def plot_ibs_curve_DT(
    DT_R,
    T, E,                         # arrays/Series aligned with DT_R['Surv'] rows
    groups=None,                  # optional (e.g., DRUG_test) for per-drug overlays
    horizon=24,                   # compute IBS on [0, horizon]
    outpath=None,                 # e.g., f"{outdir}/ibs_curve_TEST.png"
    title=None,
    show=True
):
    """
    Uses DT_R['Surv'] (n x m) and DT_R['months_grid'] to compute & plot Brier(t) and IBS.
    If `groups` is provided, overlays per-group curves as well.
    Returns dict with ibs_all, ibs_by_group, t_grid, brier_all, brier_by_group, fig, ax.
    """
    months_grid = np.asarray(DT_R["months_grid"], float)
    S = np.asarray(DT_R["Surv"], float)  # (n, m)

    # restrict to horizon
    mask = months_grid <= float(horizon)
    if not np.any(mask):
        raise ValueError(f"horizon={horizon} is below months_grid min={months_grid.min()}.")
    t_grid = months_grid[mask]
    S_for_ibs = S[:, mask]

    # overall
    T = np.asarray(T, float); E = np.asarray(E, int)
    kmc_all = km_censoring(T, E)
    ibs_all, brier_all = ibs_ipcw(T, E, S_for_ibs, t_grid, kmc=kmc_all)

    # figure
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=300)
    ax.plot(t_grid, brier_all, lw=2.4, color="black", label=f"ALL  (IBS={ibs_all:.3f})")

    # per-group overlays (optional)
    ibs_by_group, brier_by_group = {}, {}
    if groups is not None:
        groups = np.asarray(groups)
        uniq = list(pd.unique(groups))
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for i, g in enumerate(uniq):
            idx = (groups == g)
            if not np.any(idx): 
                continue
            T_g, E_g, S_g = T[idx], E[idx], S_for_ibs[idx]
            kmc_g = km_censoring(T_g, E_g)
            ibs_g, brier_g = ibs_ipcw(T_g, E_g, S_g, t_grid, kmc=kmc_g)
            color = palette[i % len(palette)]
            ax.plot(t_grid, brier_g, lw=2, color=color, label=f"{pretty_drug_name(g)}  (IBS={ibs_g:.3f})")
            ibs_by_group[str(g)] = ibs_g
            brier_by_group[str(g)] = brier_g

    # styling
    for side in ("left", "bottom", "right", "top"):
        ax.spines[side].set_linewidth(1.3); ax.spines[side].set_color("black")
    ax.set_xlabel("Months", fontsize=14, fontweight="bold")
    ax.set_ylabel("Brier score(t)", fontsize=14, fontweight="bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(13); tick.set_fontweight("bold")
    ttl = title or f"Brier(t) with IBS@{int(horizon)}m"
    ax.set_title(ttl, fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.grid(False)
    plt.tight_layout()
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    if show:
        plt.show()

    return dict(
        ibs_all=ibs_all, ibs_by_group=ibs_by_group,
        t_grid=t_grid, brier_all=brier_all, brier_by_group=brier_by_group,
        fig=fig, ax=ax
    )


# ===================== 3) Plots individual DT survival curves =====================
def plot_patients_survival_DT_split_legend(
    DT_R,
    ids=None,                    # list of patient IDs (row labels)
    indices=None,                # list of integer positions (0-based)
    index_like=None,             # pandas Index matching DT_R['Surv'] order (required if ids is used)
    r=None,                      # optional risk array for labeling
    drug=None,                   # optional drug array for labeling
    T=None, E=None,              # optional observed PFS time/event for legend text
    annotate=("median","mean"),  # any subset of {"median","mean"}; empty tuple for none
    # output
    outpath_curves_no_legend=None,  # e.g., f"{outdir}/patients_curves_only.png"
    outpath_legend_only=None,       # e.g., f"{outdir}/patients_legend_only.png"
    legend_ncol=1,
    title_curves="Predicted survival curves (selected patients)",
    show=True
):
    """
    Draw two outputs:
      (A) Curves-only figure (no legend), with optional median/mean vertical lines
      (B) Legend-only figure, where each entry includes real PFS time/event (if T/E provided)

    Returns dict with {"fig_curves": ..., "ax_curves": ..., "fig_legend": ...}
    """

    # --- pull from DT_R ---
    months_grid = np.asarray(DT_R["months_grid"], float)
    S_mat = np.asarray(DT_R["Surv"], float)                  # (n, m)
    med_all = DT_R.get("median_surv", None)                  # (n,) or None
    mean_all = DT_R.get("expected_surv", None)

    # --- choose rows ---
    if indices is not None:
        pos = list(indices); labels = [str(i) for i in pos]
    elif ids is not None:
        if index_like is None:
            raise ValueError("ids provided but index_like is None. Pass the Index that matches DT_R['Surv'] order.")
        idx_obj = pd.Index(index_like)
        pos = [int(idx_obj.get_loc(i)) for i in ids]
        labels = [str(i) for i in ids]
    else:
        raise ValueError("Provide either indices=[...] or ids=[...] with index_like=...")

    # Arrays for extra info
    r_arr    = None if r    is None else np.asarray(r, float)
    drug_arr = None if drug is None else np.asarray(drug)
    T_arr    = None if T    is None else np.asarray(T, float)
    E_arr    = None if E    is None else np.asarray(E, int)

    # Colors
    # palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
    #            "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]

    palette = ["tab:orange",
               "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]

    # ---- (A) Curves-only figure (no legend) ----
    # figA, axA = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    figA, axA = plt.subplots(figsize=(3.8, 3), dpi=300)
    line_handles, legend_labels = [], []

    for k, (p, lbl) in enumerate(zip(pos, labels)):
        S_row = S_mat[p]
        color = palette[k % len(palette)]

        # build a base label (used for legend-only fig)
        parts = [f"ID {lbl}"]
        if drug_arr is not None:
            # you already have pretty_drug_name in your utils; if not, comment next line:
            try:
                from ALK_Calibration_utils import pretty_drug_name
                parts.append(pretty_drug_name(drug_arr[p]))
            except Exception:
                parts.append(str(drug_arr[p]))
        if r_arr is not None:
            parts.append(f"r={r_arr[p]:.3f}")
        if (T_arr is not None) and (E_arr is not None):
            parts.append(f"PFS={T_arr[p]:.1f}m (event={int(E_arr[p])})")
        leg_label = " — ".join(parts)
        legend_labels.append(leg_label)

        # draw curve
        h, = axA.plot(months_grid, S_row, lw=2.5, color=color, label=leg_label)
        line_handles.append(h)

        # optional annotations
        if "median" in annotate and med_all is not None:
            med = float(med_all[p])
            if np.isfinite(med):
                axA.axvline(med, color=color, lw=1.6, ls="--", alpha=0.9)
        if "mean" in annotate and mean_all is not None:
            mea = float(mean_all[p])
            if np.isfinite(mea):
                axA.axvline(mea, color=color, lw=1.6, ls=":", alpha=0.9)

    # style: no legend in this figure
    for side in ("left", "bottom", "right", "top"):
        axA.spines[side].set_linewidth(1.4); axA.spines[side].set_color("black")
    axA.set_xlim(0, float(months_grid[-1])); axA.set_ylim(0, 1.0)

    # major tick every 12 months, optional minor every 3 months
    max_month = float(months_grid[-1])
    axA.set_xlim(0, max_month)
    axA.xaxis.set_major_locator(mticker.MultipleLocator(12))
    axA.xaxis.set_minor_locator(mticker.MultipleLocator(3))
    axA.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # no decimals

    axA.set_xlabel("Time (months)", fontsize=14, fontweight="bold")
    axA.set_ylabel("Survival probability", fontsize=14, fontweight="bold")
    for tick in axA.get_xticklabels() + axA.get_yticklabels():
        tick.set_fontsize(13); tick.set_fontweight("bold")
    axA.grid(False)
    axA.set_title(title_curves, fontsize=14, fontweight="bold")
    plt.tight_layout()
    figA.patch.set_alpha(0.0); axA.patch.set_alpha(0.0)

    if outpath_curves_no_legend:
        os.makedirs(os.path.dirname(outpath_curves_no_legend), exist_ok=True)
        figA.savefig(outpath_curves_no_legend, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    if show:
        plt.show()

    # ---- (B) Legend-only figure ----
    # (Create a blank figure and draw only the legend; use handles from the curves.)
    # Set the figure height based on number of legend rows
    n = len(legend_labels)
    fig_h = 0.6 + 0.35 * max(1, (n / legend_ncol))
    figB = plt.figure(figsize=(8, fig_h), dpi=300)
    figB.legend(handles=line_handles, labels=legend_labels,
                loc="center", ncol=legend_ncol, frameon=False, fontsize=11)
    plt.axis("off")
    figB.patch.set_alpha(0.0)

    if outpath_legend_only:
        os.makedirs(os.path.dirname(outpath_legend_only), exist_ok=True)
        figB.savefig(outpath_legend_only, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    if show:
        plt.show()

    return {"fig_curves": figA, "ax_curves": axA, "fig_legend": figB}


# ===================== 4) Estimates per-patient ITE (ΔRMST) for LCT =====================
def estimate_LCT_ITE_for_test(
    T_train, E_train, DRUG_train,
    T_test,  E_test,  DRUG_test,
    r_train_oof, r_test,
    family_noLCT="weibull",
    family_LCT="weibull",
    use_transport_weights=True,
    # ── Covariate adjustment for transport weights ─────────────────────────
    X_train_transport=None,
    # pd.DataFrame or np.ndarray, shape (n_train, p).
    # Baseline covariates used to estimate the density ratio
    # p(TEST) / p(TRAIN) via multivariate logistic regression.
    # Recommended columns (example):
    #   cols_transport = ['RECIST_before','Age','Sex',
    #                     'TV_AllTumor_after','TV_AllTumor_before',
    #                     'NLR_2','NEUT_2']
    #   X_train_transport = x_train.loc[:, cols_transport]
    # When None, falls back to the 1-D risk score r_train_oof.
    X_test_transport=None,
    # pd.DataFrame or np.ndarray, shape (n_test, p).  Same columns as
    # X_train_transport.
    #   X_test_transport = x_testBS.loc[:, cols_transport]
    # When None, falls back to the 1-D risk score r_test.
    times=(6, 12, 24),
    rmst_tau=36,
    months_max=60,
    # ── Landmark correction ────────────────────────────────────────────────
    landmark_month=None,
    # Set to a positive float (e.g. 4.0) to activate landmark analysis.
    # • Patients with T ≤ landmark are excluded (they could not have
    #   survived long enough to receive LCT).
    # • Survival times are shifted: T* = T − landmark, so t = 0 in the
    #   calibrated curves means "landmark months after systemic start".
    # • rmst_tau is interpreted on the SHIFTED scale.  If your rmst_tau
    #   was set on the original scale (e.g. 36 months from study entry),
    #   pass  rmst_tau = original_tau − landmark_month  when landmark is
    #   active, or let the function clip and warn automatically.
    # • Leave as None (default) to reproduce the original behaviour exactly.
    # ── Bootstrap 95 % CI for ΔRMST ───────────────────────────────────────
    n_boot=500,
    # Number of bootstrap replicates for per-patient 95 % CI of ΔRMST.
    # Strategy (paired):
    #   Each replicate resamples TRAIN(Drug-B) WITH replacement → new
    #   cal_noLCT_b (+ new transport weights).  TEST stays fixed so that
    #   cal_LCT is identical every replicate.  Both calibrators predict on
    #   the original TEST patients → paired ΔRMST_b per patient.
    #   CI = [2.5th, 97.5th] percentile across replicates.
    # Set to 0 to skip bootstrap (faster; df_ite will lack CI columns).
    boot_seed=42,
    # Random seed for reproducibility of the bootstrap.
):
    """
    Estimate the Individual Treatment Effect (ITE) of LCT for TEST patients.

    Setting
    -------
    • TRAIN patients received drug-1 or drug-2, WITHOUT LCT  (A = 0).
    • TEST  patients received drug-2 WITH LCT               (A = 1).
    • Goal : for each TEST patient, estimate the counterfactual survival
             had they NOT received LCT, then compute ITE = S_LCT − S_noLCT.

    Steps
    -----
    0) (Optional) Landmark correction:
          Exclude patients with T ≤ landmark_month and shift T* = T − landmark.
          This removes immortal-time bias when LCT is administered up to
          landmark_month months after systemic therapy start.
    1) Identify the unique drug label in TEST (Drug-B).
    2) Subset TRAIN to Drug-B rows → the no-LCT comparator arm.
    3) Fit transport weights so TRAIN(Drug-B, no-LCT) mimics the
       covariate distribution of TEST(Drug-B, LCT).
    4) Fit cal_noLCT on weighted TRAIN(Drug-B).
    5) Fit cal_LCT   on TEST(Drug-B).
    6) Predict S0 (counterfactual, no LCT) and S1 (factual, LCT) for every
       TEST patient.
    7) Compute ΔS(t), ΔRMST, ΔMedian, ΔMean per patient.

    Parameters
    ----------
    T_train, E_train, DRUG_train : array-like
        Survival times, event indicators, and drug labels for TRAIN patients.
    T_test, E_test, DRUG_test : array-like
        Survival times, event indicators, and drug labels for TEST patients.
        DRUG_test must contain exactly ONE unique label (Drug-B).
    r_train_oof, r_test : array-like
        Model risk scores (e.g. predicted AFT log-time) for TRAIN and TEST.
        Used as the fallback (1-D) covariate for transport weights when
        X_train_transport / X_test_transport are not provided.
    family_noLCT, family_LCT : str
        Parametric survival family for each calibrator ('weibull', 'lognormal', …).
    use_transport_weights : bool
        If True, reweight TRAIN(Drug-B) toward TEST(Drug-B) via density ratio.
    X_train_transport : pd.DataFrame | np.ndarray | None
        Baseline covariates for TRAIN patients used in multivariate transport
        weight estimation (shape n_train × p).  When provided together with
        X_test_transport, a multi-dimensional logistic density ratio is fitted;
        otherwise falls back to the 1-D risk score.
        Example: x_train.loc[:, ['RECIST_before','Age','Sex',
                                  'TV_AllTumor_after','TV_AllTumor_before',
                                  'NLR_2','NEUT_2']]
    X_test_transport : pd.DataFrame | np.ndarray | None
        Same columns as X_train_transport, for TEST patients.
        Example: x_testBS.loc[:, cols_transport]
    times : tuple of float
        Landmark times (on the shifted scale if landmark active) at which to
        report ΔS.
    rmst_tau : float
        RMST truncation time (on the shifted scale if landmark active).
    months_max : float
        Upper bound for the months grid.
    landmark_month : float or None
        Landmark time in months.  None = no landmark correction (original
        behaviour).
    n_boot : int
        Number of bootstrap replicates for 95 % CI of ΔRMST (default 500).
        Set to 0 to skip; CI columns will be absent from df_ite.
    boot_seed : int
        Random seed for the bootstrap RNG (default 42).

    Returns
    -------
    df_ite      : pd.DataFrame  — per-patient ITE columns, including
                  ΔRMST@{tau}m_CI_low / _CI_high when n_boot > 0
    curves      : dict          — S0, S1, time grid, landmark used
    calibrators : dict          — {'B_noLCT': cal, 'B_LCT': cal}
    months_grid : np.ndarray
    """
    import warnings

    # ── 0) Landmark correction  ───────────────────────────────────────────────
    # Convert all inputs to numpy first so boolean indexing works uniformly.
    T_tr = np.asarray(T_train,    float)
    E_tr = np.asarray(E_train,    int)
    D_tr = _norm(DRUG_train)
    r_tr = np.asarray(r_train_oof, float)

    T_te = np.asarray(T_test,  float)
    E_te = np.asarray(E_test,  int)
    D_te = _norm(DRUG_test)
    r_te = np.asarray(r_test,  float)

    # Convert optional covariate matrices (keep as numpy; None stays None)
    X_tr = np.asarray(X_train_transport, float) if X_train_transport is not None else None
    X_te = np.asarray(X_test_transport,  float) if X_test_transport  is not None else None

    # Preserve the original TEST index so df_ite rows align with df_plot rows.
    te_index = (DRUG_test.index if hasattr(DRUG_test, "index") else None)

    if landmark_month is not None and landmark_month > 0:
        # --- TRAIN: keep T > landmark, shift time origin ---
        keep_tr = T_tr > landmark_month
        n_drop_tr = int((~keep_tr).sum())
        if n_drop_tr > 0:
            warnings.warn(
                f"Landmark={landmark_month} mo: dropping {n_drop_tr} TRAIN "
                f"patient(s) with T ≤ {landmark_month} mo.",
                UserWarning,
            )
        T_tr = T_tr[keep_tr] - landmark_month
        E_tr = E_tr[keep_tr]
        D_tr = D_tr[keep_tr]
        r_tr = r_tr[keep_tr]
        if X_tr is not None:
            X_tr = X_tr[keep_tr]          # ← filter X in sync with T/E/r

        # --- TEST: keep T > landmark, shift time origin ---
        keep_te = T_te > landmark_month
        n_drop_te = int((~keep_te).sum())
        if n_drop_te > 0:
            warnings.warn(
                f"Landmark={landmark_month} mo: dropping {n_drop_te} TEST "
                f"patient(s) with T ≤ {landmark_month} mo.",
                UserWarning,
            )
        T_te = T_te[keep_te] - landmark_month
        E_te = E_te[keep_te]
        D_te = D_te[keep_te]
        r_te = r_te[keep_te]
        if X_te is not None:
            X_te = X_te[keep_te]          # ← filter X in sync with T/E/r

        # Preserve aligned index for df_ite
        if te_index is not None:
            te_index = te_index[keep_te]

        lm_used = landmark_month
    else:
        lm_used = None   # landmark not applied; record in curves dict

    # ── 1) Build months grid on (possibly shifted) times ─────────────────────
    months_grid = build_months_grid(T_tr, T_te, months_max=months_max)

    # Clip rmst_tau to the observable horizon (important after landmark shift)
    max_horizon = float(months_grid[-1])
    if rmst_tau > max_horizon:
        warnings.warn(
            f"rmst_tau={rmst_tau} exceeds the{'  landmark-shifted' if lm_used else ''} "
            f"grid max ({max_horizon:.1f} mo).  Clipping to {max_horizon:.1f}.",
            UserWarning,
        )
        rmst_tau = max_horizon

    # ── 2) Identify the single drug label in TEST ─────────────────────────────
    uniq_test = pd.unique(D_te)
    if len(uniq_test) != 1:
        raise ValueError(
            f"TEST must have a single drug label; "
            f"found {list(pd.unique(_norm(DRUG_test)))}."
        )
    drugB_norm = uniq_test[0]

    # ── 3) Subset TRAIN to Drug-B (no-LCT comparator) ────────────────────────
    idx_B = (D_tr == drugB_norm)
    if not np.any(idx_B):
        raise ValueError(
            "No TRAIN rows match the TEST drug label "
            f"('{drugB_norm}'). Cannot estimate LCT effect."
        )
    r_tr_B = r_tr[idx_B]
    T_tr_B = T_tr[idx_B]
    E_tr_B = E_tr[idx_B]
    X_tr_B = X_tr[idx_B] if X_tr is not None else None   # ← subset covariates

    # ── 4) Transport weights: reweight TRAIN(Drug-B, noLCT) → TEST(Drug-B, LCT)
    # Priority: multivariate X  >  1-D risk score  >  uniform (on failure)
    w_tr = None
    if use_transport_weights and len(r_tr_B) > 0 and len(r_te) > 0:
        try:
            if X_tr_B is not None and X_te is not None:
                # Multivariate adjustment using provided baseline covariates
                w_tr = density_ratio_logit_multifeature(X_tr_B, X_te)
            else:
                # Fallback: 1-D density ratio on the model risk score
                w_tr = density_ratio_logit(r_tr_B, r_te)
        except Exception as exc:
            warnings.warn(
                f"Transport weight estimation failed ({exc}); "
                "using uniform weights.",
                UserWarning,
            )
            w_tr = None

    # ── 5) Fit calibrators ────────────────────────────────────────────────────
    cal_noLCT = WeightedParametricCalibrator(family_noLCT).fit(
        r_tr_B, T_tr_B, E_tr_B, weights=w_tr
    )
    cal_LCT = WeightedParametricCalibrator(family_LCT).fit(
        r_te, T_te, E_te, weights=None
    )

    # ── 6) Predict survival curves for all TEST patients ─────────────────────
    # S0 : counterfactual — what would have happened WITHOUT LCT
    # S1 : factual        — calibrated on the observed LCT experience
    S0 = cal_noLCT.predict_survival_at_times(r_te, months_grid)
    S1 = cal_LCT.predict_survival_at_times(  r_te, months_grid)

    def S_at(Smat, t):
        """Return S(t) for all patients via direct lookup or interpolation."""
        if np.any(np.isclose(months_grid, t)):
            j = int(np.argwhere(np.isclose(months_grid, t)).ravel()[0])
            return Smat[:, j]
        return np.array([np.interp(t, months_grid, Smat[i])
                         for i in range(Smat.shape[0])])

    # ── 7) Per-patient ITE summaries ──────────────────────────────────────────
    n_te = len(r_te)
    cols = {}

    for t in times:
        cols[f"S_noLCT@{t}m"] = S_at(S0, t)
        cols[f"S_LCT@{t}m"]   = S_at(S1, t)
        cols[f"ΔS@{t}m"]      = cols[f"S_LCT@{t}m"] - cols[f"S_noLCT@{t}m"]

    RM0 = np.array([rmst_from_curve(months_grid, S0[i], rmst_tau) for i in range(n_te)])
    RM1 = np.array([rmst_from_curve(months_grid, S1[i], rmst_tau) for i in range(n_te)])
    cols[f"RMST_noLCT@{rmst_tau}m"] = RM0
    cols[f"RMST_LCT@{rmst_tau}m"]   = RM1
    cols[f"ΔRMST@{rmst_tau}m"]      = RM1 - RM0

    try:
        med0 = cal_noLCT.predict_median(r_te)
        med1 = cal_LCT.predict_median(r_te)
        cols["Median_noLCT"] = med0
        cols["Median_LCT"]   = med1
        cols["ΔMedian"]      = med1 - med0
    except Exception:
        pass

    try:
        mean0 = cal_noLCT.predict_expectation(r_te)
        mean1 = cal_LCT.predict_expectation(r_te)
        cols["Mean_noLCT"] = mean0
        cols["Mean_LCT"]   = mean1
        cols["ΔMean"]      = mean1 - mean0
    except Exception:
        pass

    df_ite = pd.DataFrame(cols, index=te_index)
    df_ite["Preferred"] = np.where(
        df_ite[f"ΔRMST@{rmst_tau}m"] > 0, "LCT", "No LCT"
    )

    # ── 8) Bootstrap 95 % CI for per-patient ΔRMST ───────────────────────────
    # Strategy (paired bootstrap):
    #   • TRAIN(Drug-B) is resampled WITH replacement each replicate.
    #     → new cal_noLCT_b (+ refreshed transport weights)
    #     → new S0_b (counterfactual) for all TEST patients
    #   • TEST stays FIXED every replicate.
    #     → cal_LCT is identical to the point-estimate cal_LCT (no noise added)
    #     → S1 is identical to the point-estimate S1
    #   • ΔRMST_b = RMST(S1) − RMST(S0_b)   [only S0 varies]
    #   • CI = [2.5th, 97.5th] percentile of ΔRMST_b across replicates.
    #
    # Rationale: the LCT arm (TEST) has a fixed, fully-observed outcome; its
    # uncertainty comes from the noLCT calibrator estimated on TRAIN.
    # Resampling only TRAIN propagates that uncertainty without disturbing the
    # factual arm — giving a valid paired interval for the causal contrast.
    if n_boot and n_boot > 0:
        rng         = np.random.default_rng(boot_seed)
        boot_deltas = np.zeros((n_boot, n_te), float)  # shape (B, n_test)

        for b in range(n_boot):
            # --- resample TRAIN(Drug-B) with replacement ---
            idx_b   = rng.integers(0, len(r_tr_B), size=len(r_tr_B))
            r_tr_b  = r_tr_B[idx_b]
            T_tr_b  = T_tr_B[idx_b]
            E_tr_b  = E_tr_B[idx_b]

            # --- transport weights for this TRAIN replicate vs. fixed TEST ---
            w_b = None
            if use_transport_weights and len(r_tr_b) > 0:
                try:
                    X_tr_b = X_tr_B[idx_b] if X_tr_B is not None else None
                    if X_tr_b is not None and X_te is not None:
                        w_b = density_ratio_logit_multifeature(X_tr_b, X_te)
                    else:
                        w_b = density_ratio_logit(r_tr_b, r_te)
                except Exception:
                    w_b = None

            # --- new noLCT calibrator ---
            cal_noLCT_b = WeightedParametricCalibrator(family_noLCT).fit(
                r_tr_b, T_tr_b, E_tr_b, weights=w_b
            )

            # --- counterfactual curves on original TEST patients ---
            S0_b = cal_noLCT_b.predict_survival_at_times(r_te, months_grid)

            # --- ΔRMST for this replicate (S1 is the fixed point-estimate) ---
            RM0_b = np.array([rmst_from_curve(months_grid, S0_b[i], rmst_tau)
                               for i in range(n_te)])
            boot_deltas[b] = RM1 - RM0_b   # RM1 from point estimate (fixed)

        ci_lo = np.quantile(boot_deltas, 0.025, axis=0)
        ci_hi = np.quantile(boot_deltas, 0.975, axis=0)

        df_ite[f"ΔRMST@{rmst_tau}m_CI_low"]  = ci_lo
        df_ite[f"ΔRMST@{rmst_tau}m_CI_high"] = ci_hi

        # Significance flag: CI excludes zero on both sides
        df_ite[f"ΔRMST@{rmst_tau}m_sig"] = (
            (ci_lo > 0) | (ci_hi < 0)
        )

    calibrators = {"B_noLCT": cal_noLCT, "B_LCT": cal_LCT}
    curves = {
        "S_noLCT":  S0,
        "S_LCT":    S1,
        "t":        months_grid,
        "landmark": lm_used,
    }

    return df_ite, curves, calibrators, months_grid


# ===================== 5) Individual counterfactual survival curves =====================
def plot_lct_counterfactual_for_patient(calibrators_B, r_value, months_grid,
                                        observed_time=None, observed_event=None,
                                        title=None, outpath=None, show=True,
                                        legend_outpath=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # survival curves
    cal0, cal1 = calibrators_B["B_noLCT"], calibrators_B["B_LCT"]
    r_vec = np.array([float(r_value)], float)
    S0 = cal0.predict_survival_at_times(r_vec, months_grid)[0]
    S1 = cal1.predict_survival_at_times(r_vec, months_grid)[0]

    # main plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    l1, = ax.plot(months_grid, S1, lw=2.5, label="With LCT", color="tab:blue")
    l0, = ax.plot(months_grid, S0, lw=2.5, label="No LCT",  color="tab:orange")
    if observed_time is not None:
        ls = "-" if (observed_event == 0) else "--"
        obs_line = ax.axvline(round(float(observed_time)), color="black", lw=1.6, ls=ls,
                              label=f"Observed PFS={round(observed_time)} months (event={int(observed_event)})")
    else:
        obs_line = None

    # formatting
    for s in ("left","bottom","right","top"):
        ax.spines[s].set_linewidth(1.3); ax.spines[s].set_color("black")
    ax.set_xlim(0, float(months_grid[-1])); ax.set_ylim(0, 1.0)

    # major tick every 12 months, optional minor every 3 months
    max_month = float(months_grid[-1])
    ax.set_xlim(0, max_month)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(12))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(3))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # no decimals

    ax.set_xlabel("Time (months)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Survival probability", fontsize=13, fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(12); t.set_fontweight("bold")
    ax.grid(False)
    ax.set_title(title or "Counterfactual survival with vs without LCT", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)

    # --- save main plot ---
    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)

    # --- create separate legend figure ---
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig_leg = plt.figure(figsize=(4, 1.2), dpi=300)
        fig_leg.legend(handles, labels, loc="center", ncol=len(labels),
                       frameon=False, fontsize=11)
        plt.tight_layout()
        if legend_outpath:
            fig_leg.savefig(legend_outpath, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
        if show:
            fig_leg.show()

    if show:
        plt.show()

    return {"fig": fig, "ax": ax, "legend_fig": fig_leg if handles else None}


