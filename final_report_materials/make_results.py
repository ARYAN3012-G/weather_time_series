"""
Results tables (LaTeX) and figures for the final report, built only from saved result files
(no retraining).

    python final_report_materials/make_results.py
Writes latex/tables/*.tex and latex/figures/res_*.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W = ROOT / "results" / "v4"
S = ROOT / "satellite_module" / "results"
R = ROOT / "rabi_module"
TAB = HERE / "latex" / "tables"
FIG = HERE / "latex" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
NAVY, BLUE, ORANGE, GREY, RED, GREEN, PURPLE = "#1F3A5F", "#4F81BD", "#E08A1E", "#9A9A9A", "#C0392B", "#2E7D32", "#6E4A9E"
SITE = {"Som": "Somandepalle", "Lep": "Lepakshi", "Cho": "Cholasamudram", "Sira": "Sira", "Ped": "Peddakadubur",
        "Din": "Kurnool-Dinnedevarapadu", "Bal": "Kurnool-Balaji Villas", "Pul": "Kurnool-Pullaiah"}
SITE_ORDER = ["Som", "Lep", "Cho", "Sira", "Ped", "Din", "Bal", "Pul"]


def tex(s) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#"))


def write_table(name, caption, label, header, rows, colspec, size="\\small", note=None, bold_rows=()):
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
             "\\midrule"]
    for i, r in enumerate(rows):
        if r == "MID":
            lines.append("\\midrule")
            continue
        cells = [re.sub(r"(?<!\\)(\d)%", r"\1\\,\\%", str(c)) for c in r]
        if i in bold_rows:
            cells = [f"\\textbf{{{c}}}" if c else c for c in cells]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\\\[2pt]", f"{{\\footnotesize {note}}}"]
    lines += ["\\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print("table", name)



def _rgb(path):
    """Drop the (unused) alpha channel: identical pixels, much faster for pdfLaTeX."""
    from PIL import Image
    im = Image.open(path)
    if im.mode == "RGBA":
        im.convert("RGB").save(path, dpi=im.info.get("dpi", (300, 300)))

def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    _rgb(FIG / f"{name}.png")
    plt.close(fig)
    print("figure", name)


def pm(m, s, d=3):
    if pd.isna(m):
        return "--"
    return f"{m:.{d}f}" if pd.isna(s) or s == 0 else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


# =================================================================================================
# WEATHER
# =================================================================================================
def weather():
    s = pd.read_csv(W / "tables" / "summary_temporal.csv")
    H = [7, 15, 30]
    hyb = "weather_spei_doy_wb"

    def get(model, fs="-", h=7, col="macro_f1_mean"):
        q = s[(s.model == model) & (s.feature_set == fs) & (s.horizon == h)]
        return q[col].iloc[0] if len(q) else np.nan

    # --- models x horizons (hybrid features) + baselines
    names = {"persistence": "Persistence", "markov": "Markov transition", "climatology": "Climatology",
             "majority": "Majority class", "wb_ensemble": "Water-balance ensemble (physics)",
             "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM", "transformer": "Transformer",
             "ensemble": "Soft-voting ensemble"}
    rows = []
    for m in ["persistence", "markov", "climatology", "majority", "wb_ensemble"]:
        rows.append([names[m]] + [f"{get(m, '-', h):.3f}" for h in H])
    rows.append("MID")
    for m in ["rf", "xgb", "lstm", "transformer", "ensemble"]:
        rows.append([names[m]] + [pm(get(m, hyb, h), get(m, hyb, h, "macro_f1_std")) for h in H])
    write_table("w_models", "Weather module: \\mf{} of every model with the hybrid inputs and of every baseline "
                "(test 2019--2026, all sites; mean $\\pm$ s.d.\\ over 5 seeds).", "tab:w_models",
                ["Model", "7 days", "15 days", "30 days"], rows, "@{}lccc@{}", bold_rows=(len(rows) - 1,))

    # --- feature sets (ensemble and best single model)
    fsn = {"weather": "Weather only (9 AgERA5 variables)", "spei_only": "SPEI history only",
           "weather_spei": "Weather + SPEI history", "weather_spei_doy": "+ day of year",
           "weather_spei_doy_wb": "+ water-balance probabilities (hybrid)"}
    rows = []
    for fs, n in fsn.items():
        r = [n]
        for h in H:
            q = s[(s.feature_set == fs) & (s.horizon == h) & (s.kind == "model")]
            if not len(q):
                r.append("--")
                continue
            ens = q[q.model == "ensemble"].macro_f1_mean.iloc[0]
            r.append(f"{ens:.3f}")
        rows.append(r)
    rows.append(["Persistence (reference)"] + [f"{get('persistence', '-', h):.3f}" for h in H])
    write_table("w_features", "Weather module: effect of the input feature set (\\mf{} of the soft-voting ensemble, "
                "test 2019--2026).", "tab:w_features", ["Inputs", "7 days", "15 days", "30 days"], rows,
                "@{}lccc@{}", bold_rows=(4,))

    # --- severe drought / probabilistic
    rows = []
    for h in H:
        rows.append([f"{h} days", f"{get('ensemble', hyb, h, 'severe_f1_mean'):.3f}",
                     f"{get('ensemble', hyb, h, 'tuned_severe_pod_mean'):.2f}",
                     f"{get('ensemble', hyb, h, 'tuned_severe_far_mean'):.2f}",
                     f"{get('ensemble', hyb, h, 'tuned_severe_csi_mean'):.2f}",
                     f"{get('ensemble', hyb, h, 'roc_auc_macro_ovr_mean'):.3f}",
                     f"{get('ensemble', hyb, h, 'bss_vs_climatology_mean'):+.2f}",
                     f"{get('persistence', '-', h, 'severe_f1_mean'):.3f}"])
    write_table("w_severe", "Weather module: severe-drought warning and probabilistic skill of the hybrid ensemble "
                "(test 2019--2026). POD / FAR / CSI at a severe-probability threshold tuned on validation; "
                "BSS = Brier skill score vs climatology.", "tab:w_severe",
                ["Horizon", "Severe F1", "POD", "FAR", "CSI", "ROC-AUC", "BSS", "Severe F1 persistence"], rows,
                "@{}lccccccc@{}", size="\\footnotesize")

    # --- LOCO
    l = pd.read_csv(W / "tables" / "summary_loco.csv")
    rows = []
    for fold, nm in (("test_kurnool", "Train south $\\rightarrow$ test Kurnool"),
                     ("test_south", "Train Kurnool $\\rightarrow$ test south")):
        for m, fs, mn in (("persistence", "-", "Persistence"), ("wb_ensemble", "-", "Water-balance ensemble"),
                          ("ensemble", "weather_spei", "Ensemble, weather + SPEI"),
                          ("ensemble", hyb, "Ensemble, hybrid")):
            q = l[(l.fold == fold) & (l.model == m) & (l.feature_set == fs)]
            rows.append([nm if mn == "Persistence" else "", mn] +
                        [f"{q[q.horizon == h].macro_f1_mean.iloc[0]:.3f}" for h in H])
        rows.append("MID")
    rows = rows[:-1]
    write_table("w_loco", "Weather module, leave-one-cluster-out: trained on one region, tested on the other region's "
                "2019--2026 data (\\mf).", "tab:w_loco", ["Test region", "Model", "7 days", "15 days", "30 days"],
                rows, "@{}llccc@{}", size="\\footnotesize")

    # --- Marusov protocol
    m = pd.read_csv(W / "marusov_protocol" / "median_scores.csv")
    rows = []
    for inp, mod, nm in (("spei_only", "persistence", "Persistence"),
                         ("spei_only", "logreg_last", "Logistic regression on last SPEI"),
                         ("spei_only", "xgb", "XGBoost, SPEI history"), ("spei_only", "lstm", "LSTM, SPEI history"),
                         ("weather_spei", "xgb", "XGBoost, weather + SPEI"),
                         ("weather_spei", "lstm", "LSTM, weather + SPEI")):
        r = [nm]
        for h in H:
            q = m[(m.horizon == h) & (m.input == inp) & (m.model == mod)]
            r.append(f"{q.median_roc_auc_mean.iloc[0]:.3f}")
        rows.append(r)
    write_table("w_marusov", "Weather module under the Marusov et al.\\ protocol (binary drought SPEI $< -1$; median "
                "ROC-AUC over sites).", "tab:w_marusov", ["Model and input", "7 days", "15 days", "30 days"], rows,
                "@{}lccc@{}", size="\\footnotesize", bold_rows=(4,))

    # --- confusion matrices
    files = {7: "confusion_temporal_all_sites_h7_weather_spei_doy_wb_xgb.csv",
             15: "confusion_temporal_all_sites_h15_weather_spei_doy_wb_lstm.csv",
             30: "confusion_temporal_all_sites_h30_weather_spei_doy_wb_lstm.csv"}
    mods = {7: "XGBoost", 15: "LSTM", 30: "LSTM"}
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, h in zip(axs, H):
        c = pd.read_csv(W / "tables" / files[h], index_col=0).to_numpy(float)
        n = c / c.sum(1, keepdims=True)
        ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{n[i, j]:.0%}\n({int(c[i, j]):,})", ha="center", va="center", fontsize=8,
                        color="white" if n[i, j] > 0.55 else "#1E1E1E")
        lab = ["Healthy", "Moderate", "Severe"]
        ax.set_xticks(range(3), lab)
        ax.set_yticks(range(3), lab)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if h == 7 else "")
        ax.set_title(f"{h} days ahead ({mods[h]}, hybrid inputs)", fontsize=9)
        ax.grid(False)
    fig.tight_layout()
    savefig(fig, "res_w_confusion")

    # --- bar chart of feature sets x horizon (ensemble) with persistence and physics
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    series = [("Persistence", [get("persistence", "-", h) for h in H], "#C8C8C8"),
              ("Physics only", [get("wb_ensemble", "-", h) for h in H], "#9DC3E6")]
    for fs, col in (("weather", "#F4B183"), ("weather_spei", "#A9D18E"), ("weather_spei_doy_wb", NAVY)):
        series.append(({"weather": "Weather only", "weather_spei": "Weather + SPEI",
                        "weather_spei_doy_wb": "Hybrid"}[fs], [get("ensemble", fs, h) for h in H], col))
    x = np.arange(3)
    bw = 0.16
    for i, (nm, v, col) in enumerate(series):
        b = ax.bar(x + (i - 2) * bw, v, bw, label=nm, color=col, edgecolor="white")
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [f"{h} days ahead" for h in H])
    ax.set_ylabel("Macro-F1 (test 2019-2026)")
    ax.set_ylim(0, 0.95)
    ax.legend(ncol=5, fontsize=7.5, loc="upper right", frameon=False)
    savefig(fig, "res_w_bars")


# =================================================================================================
# SATELLITE
# =================================================================================================
def satellite():
    # --- lag curves
    lc = pd.read_csv(S / "lag" / "lag_correlation.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, irr, title in ((axs[0], 0, "Rainfed fields"), (axs[1], 1, "Irrigated fields")):
        for sc, col in ((30, NAVY), (60, ORANGE), (90, GREEN)):
            q = lc[(lc.irrigated == irr) & (lc.spei_scale == sc)].groupby("lag_days").rho.median()
            ax.plot(q.index, q.values, marker="o", ms=3, color=col, label=f"SPEI-{sc}")
            i = q.values.argmax()
            ax.scatter([q.index[i]], [q.values[i]], s=45, color=col, zorder=5, edgecolor="white")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Lag: crop SVI on day t vs SPEI on day t - lag (days)")
        fl, fs_ = (10, 30) if irr == 0 else (15, 60)
        ax.axvline(fl, color=GREY, ls=":", lw=1)
        ax.text(fl + 1.5, 0.05, f"frozen choice\n(SPEI-{fs_}, {fl} d,\nchosen on 2019-23)", fontsize=7, color=GREY)
    axs[0].set_ylabel("Spearman ρ (median over sites, 2019-2026)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "res_s_lag")

    fz = pd.read_csv(S / "lag" / "frozen_lag.csv")
    rows = [["Rainfed" if r.irrigated == 0 else "Irrigated", f"SPEI-{r.spei_scale}", f"{r.lag_days} days",
             f"{r.rho_train_median:.2f}", f"{r.rho_test_median:.2f} ({r.rho_test_min:.2f}--{r.rho_test_max:.2f})"]
            for r in fz.itertuples()]
    write_table("s_lag", "Weather-to-crop lag: SPEI scale and lag chosen on 2019--2023 only, then measured on the "
                "unseen years 2024--2026 (Spearman $\\rho$ between crop SVI and lagged SPEI, median over sites).",
                "tab:s_lag", ["Fields", "SPEI scale", "Lag", "$\\rho$ 2019--2023", "$\\rho$ 2024--2026 (range)"],
                rows, "@{}lcccc@{}", size="\\footnotesize")

    # --- alarm confirmation per site
    cf = pd.read_csv(S / "lag" / "confirmation.csv")
    cf = cf[cf.source == "median of sampled fields"].set_index("site").reindex(SITE_ORDER)
    rows = []
    for st, r in cf.iterrows():
        rows.append([SITE[st], int(r.alarms), int(r.confirmed), f"{r.confirmation_rate:.0%}",
                     f"{r.base_rate_any_date:.0%}",
                     "--" if pd.isna(r.median_delay_days) else f"{r.median_delay_days:.0f}"])
    rows.append(["\\textbf{All sites}", int(cf.alarms.sum()), int(cf.confirmed.sum()),
                 f"{cf.confirmed.sum() / cf.alarms.sum():.0%}", "", ""])
    write_table("s_confirm", "Satellite module: weather alarms (SPEI-30 $< -1$, 2019--2026) followed by rainfed crop "
                "stress (SVI $< -1$) within 60 days, compared with the share of all dates that are stressed "
                "(base rate).", "tab:s_confirm",
                ["Site", "Alarms", "Confirmed", "Rate", "Base rate", "Median delay (d)"], rows,
                "@{}lccccc@{}", size="\\footnotesize")

    # --- forecast versions, RF and XGB, 15 and 30 days, with onset AUC
    f = pd.read_csv(S / "forecast" / "summary_temporal.csv")
    allf = f[f.fold == "all_sites"]
    vn = {"baseline/persistence": "Persistence", "satellite": "1  Satellite history",
          "sat_pastweather": "2  + past weather", "sat_weatherforecast": "3  + weather-module forecast",
          "sat_futureweather_UPPERBOUND": "4  + actual future weather (upper bound)", "weather_only": "Weather only"}
    rows = []
    for key, nm in vn.items():
        r = [nm]
        for h in (15, 30):
            for mdl in ("rf", "xgb"):
                k = key if key.startswith("baseline") else f"{key}/{mdl}"
                q = allf[(allf.model == k) & (allf.horizon == h)]
                r.append(f"{q.macro_f1.iloc[0]:.3f}" if len(q) else "--")
            q = allf[(allf.model == (key if key.startswith("baseline") else f"{key}/xgb")) & (allf.horizon == h)]
            r.append(f"{q.onset_roc_auc.iloc[0]:.2f}" if len(q) else "--")
        rows.append(r)
    for nm, k in (("Logistic baseline on SVI level + trend", "onset-baseline/svi_logit"),):
        r = [nm]
        for h in (15, 30):
            q = allf[(allf.model == k) & (allf.horizon == h)]
            r += ["", "", f"{q.onset_roc_auc.iloc[0]:.2f}" if len(q) else "--"]
        rows.append(r)
    write_table("s_forecast", "Satellite module: crop-condition (SVI class) forecast on test years 2025--2026 "
                "(\\mf, mean of 5 seeds) and onset warning (ROC-AUC for fields normal today that fall below normal "
                "by the target date).", "tab:s_forecast",
                ["Version", "RF 15 d", "XGB 15 d", "Onset AUC 15 d", "RF 30 d", "XGB 30 d", "Onset AUC 30 d"],
                [[tex(r[0]).replace("  ", "\\; ")] + r[1:] for r in rows], "@{}lcccccc@{}", size="\\footnotesize")

    # --- subsets (15 days)
    subsets = [("all_sites", "All fields"), ("rainfed", "Rainfed fields"), ("irrigated", "Irrigated fields"),
               ("kharif", "Kharif targets (Jun--Nov)"), ("rabi", "Rabi targets (Dec--Mar)"),
               ("2026", "2026 drought season (Jul--Nov)")]
    rows = []
    for key, nm in subsets:
        q = f[(f.horizon == 15) & (f.fold.str.contains(key) if key != "all_sites" else f.fold == "all_sites")]
        if not len(q):
            continue
        g = lambda k: q[q.model == k].macro_f1.iloc[0] if len(q[q.model == k]) else np.nan  # noqa: E731
        o = lambda k: q[q.model == k].onset_roc_auc.iloc[0] if len(q[q.model == k]) else np.nan  # noqa: E731
        rows.append([nm, f"{g('baseline/persistence'):.3f}", f"{g('satellite/rf'):.3f}", f"{g('satellite/xgb'):.3f}",
                     f"{g('sat_weatherforecast/xgb'):.3f}", f"{g('weather_only/xgb'):.3f}",
                     f"{o('sat_weatherforecast/xgb'):.2f}"])
    write_table("s_subsets", "Satellite module: 15-day crop-condition forecast by field type and season "
                "(test 2025--2026, \\mf; last column onset ROC-AUC of version 3).", "tab:s_subsets",
                ["Subset", "Persistence", "Sat.\\ RF", "Sat.\\ XGB", "+ wx forecast XGB", "Weather only",
                 "Onset AUC"], rows, "@{}lcccccc@{}", size="\\footnotesize")

    # --- replay per site and season (kharif)
    rp = pd.read_csv(S / "replay" / "replay_seasons.csv")
    rp = rp[rp.kind == "kharif"]

    def cell(r):
        off = str(r.official)
        if "/" in off or "(" in off and "likely" not in off:
            return "ambiguous mandal"
        conf = pd.notna(r.first_confirmed)
        if off.startswith(("Severe", "Moderate")):
            lvl = off.split()[0]
            if not conf:
                return f"{lvl}: missed"
            return f"{lvl}: {int(r.lead_days_before_GO)} d before" if r.confirmed_before_GO else f"{lvl}: after order"
        if off.startswith("Not declared"):
            return "none: \\textit{false alarm}" if conf else "none: quiet"
        if "likely" in off:
            return "likely declared: confirmed" if conf else "likely declared"
        return "no order found" + (": confirmed" if conf else "")
    rows = []
    for st in SITE_ORDER:
        r = [SITE[st]]
        for y in (2023, 2024, 2025, 2026):
            q = rp[(rp.site == st) & (rp.year == y)]
            r.append(cell(q.iloc[0]) if len(q) else "--")
        rows.append(r)
    write_table("s_replay_sites", "Operational replay per site and kharif season: official declaration and system "
                "result (``2 d before'' = confirmed 2 days before the government order). 2023 retrospective, "
                "2024--2026 prospective; 2026 data to 24 October.", "tab:s_replay_sites",
                ["Site", "2023", "2024", "2025", "2026"], rows, "@{}lllll@{}", size="\\footnotesize")

    # --- case-study timelines from the replay
    rd = pd.read_csv(S / "replay" / "replay_daily.csv", parse_dates=["issue_date"])
    cases = [("Bal", 2023), ("Lep", 2026), ("Din", 2026), ("Som", 2025)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2), sharey=False)
    for ax, (st, y) in zip(axs.flat, cases):
        q = rd[(rd.site == st) & (rd.year == y) & (rd.kind == "kharif")].sort_values("issue_date")
        meta = rp[(rp.site == st) & (rp.year == y)].iloc[0]
        ax.plot(q.issue_date, q.SPEI30, color=BLUE, lw=1.6, label="SPEI-30 (weather, 7 d late)")
        ax.plot(q.issue_date, q.median_SVI, color=GREEN, lw=1.6, label="Median rainfed SVI (satellite)")
        ax.axhline(-1, color=GREY, ls="--", lw=0.8)
        al = q[q.alarm]
        ax.scatter(al.issue_date, np.full(len(al), 2.35), marker="|", s=60, color=ORANGE, label="Weather alarm")
        cf_ = q[q.confirmed]
        ax.scatter(cf_.issue_date, cf_.median_SVI, s=26, color=RED, zorder=5, label="Confirmed")
        if isinstance(meta.GO_date, str) and meta.GO_date:
            ax.axvline(pd.Timestamp(meta.GO_date), color="#1E1E1E", lw=1.2)
            ax.text(pd.Timestamp(meta.GO_date), -2.75, " G.O.", fontsize=7.5, va="bottom")
        ax.set_ylim(-3, 2.6)
        ax.set_title(f"{SITE[st]}, kharif {y} - official: {meta.official}", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylabel("Index (standard deviations)")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    savefig(fig, "res_s_cases")

    # --- drought season evidence table (2023 and 2026)
    ds = pd.read_csv(S / "events" / "drought_seasons.csv")
    rows = []
    for y in (2023, 2025, 2026):
        for st in SITE_ORDER:
            q = ds[(ds.site == st) & (ds.year == y)]
            if not len(q):
                continue
            r = q.iloc[0]
            rows.append([str(y) if st == "Som" else "", SITE[st], f"{r['rain_vs_normal_%']:+.0f}\\,\\%",
                         f"{r.SPEI30_aug_oct:+.2f}", f"{r.SVI_rainfed_aug_nov:+.2f}",
                         f"{r['rainfed_fields_below_normal_%']:.0f}\\,\\%", tex(r.official_declaration)])
        rows.append("MID")
    rows = rows[:-1]
    write_table("s_seasons", "Kharif season evidence per site: June--October rain vs normal, mean SPEI-30 and "
                "rainfed SVI in August--November, share of rainfed fields below normal, and the official order.",
                "tab:s_seasons", ["Year", "Site", "Rain", "SPEI-30", "SVI rainfed", "Fields below normal",
                                  "Official"], rows, "@{}llccccl@{}", size="\\scriptsize")

    # --- UAV vs satellite
    u = pd.read_csv(S / "uav" / "uav_points_satellite.csv")
    rows = []
    for _, r in u.iterrows():
        uv = re.sub(r"\(.*?\)", "", str(r.uav_verdict)).strip()
        sv = re.sub(r"\(.*?\)", "", str(r.satellite_verdict)).strip()
        rows.append([tex(r.point.replace("UAV_", "").replace("_", " ").title()), tex(r.crop), tex(uv), tex(sv),
                     "--" if pd.isna(r.SVI) else f"{r.SVI:+.2f}"])
    write_table("s_uav", "UAV field verdicts vs the Sentinel-2 verdict at the flight date.", "tab:s_uav",
                ["UAV field", "Crop", "UAV verdict", "Satellite verdict", "SVI"], rows, "@{}lllll@{}",
                size="\\footnotesize")


# =================================================================================================
# RABI
# =================================================================================================
def rabi():
    res = R / "results" / "forecast_v1_A1"
    met = pd.read_csv(res / "metrics.csv")
    best = ["MAE Persistence", "MAE Climatology", "MAE Anomaly persistence"]

    rows = []
    for pop, pn in (("all", "All eligible points"), ("established", "Established at issue")):
        for season in ("2023-24", "2024-25", "2025-26", "pooled"):
            for cls in ("rainfed", "irrigated", "all"):
                q = met[(met.population == pop) & (met.season == season) & (met.cls == cls)].iloc[0]
                b = q[best].astype(float)
                rows.append([pn if (season == "2023-24" and cls == "rainfed") else "",
                             season if cls == "rainfed" else "", cls, f"{int(q.n):,}", f"{q['MAE XGBoost']:.4f}",
                             f"{b.min():.4f}", tex(b.idxmin().replace("MAE ", "")),
                             f"{100 * (1 - q['MAE XGBoost'] / b.min()):.0f}\\,\\%"])
        rows.append("MID")
    rows = rows[:-1]
    write_table("r_breakdown", "Rabi forecast by test season, field type and population (MAE of NDVI 15 days ahead).",
                "tab:r_breakdown", ["Population", "Season", "Class", "Forecasts", "XGBoost", "Best baseline",
                                    "Which baseline", "Lower error"], rows, "@{}lllrcccr@{}", size="\\scriptsize")

    rows = []
    for pop, pn in (("all", "All eligible"), ("established", "Established")):
        q = met[(met.population == pop) & (met.season == "pooled") & (met.cls == "all")].iloc[0]
        rows.append([pn] + [f"{q[c]:.4f}" for c in ("MAE XGBoost", "MAE XGBoost satellite-only",
                                                      "MAE XGBoost weather 1-day (sensitivity)", "MAE Persistence",
                                                      "MAE Climatology", "MAE Anomaly persistence")])
    write_table("r_versions", "Rabi forecast: pooled MAE of all model versions and baselines (test 2023-24 to "
                "2025-26).", "tab:r_versions",
                ["Population", "XGBoost (main)", "Satellite only", "Weather 1 d late", "Persistence", "Climatology",
                 "Anomaly pers."], rows, "@{}lcccccc@{}", size="\\footnotesize")

    dm = pd.read_csv(res / "metrics_dec_mar.csv")
    rows = []
    for season in ("2023-24", "2024-25", "2025-26", "pooled"):
        q = dm[(dm.population == "all") & (dm.season == season) & (dm.cls == "all")].iloc[0]
        b = q[best].astype(float)
        rows.append([season, f"{int(q.n):,}", f"{q['MAE XGBoost']:.4f}", f"{b.min():.4f}",
                     f"{100 * (1 - q['MAE XGBoost'] / b.min()):.0f}\\,\\%"])
    write_table("r_decmar", "Rabi forecast, December--March issue dates only.", "tab:r_decmar",
                ["Season", "Forecasts", "XGBoost", "Best baseline", "Lower error"], rows, "@{}lrccr@{}",
                size="\\footnotesize")

    inst = pd.read_csv(R / "data" / "forecast" / "instances.csv.gz", usecols=["season", "status"])
    st = inst.groupby(["season", "status"]).size().unstack(fill_value=0)
    rows = []
    for y, r in st.iterrows():
        tot = r.sum()
        rows.append([f"{y}-{str(y + 1)[2:]}", f"{tot:,}", f"{r.no_recent_obs:,}", f"{r.no_target:,}", f"{r.ok:,}",
                     f"{r.ok / tot:.0%}"])
    t = st.sum()
    rows.append(["\\textbf{All}", f"{t.sum():,}", f"{t.no_recent_obs:,}", f"{t.no_target:,}", f"{t.ok:,}",
                 f"{t.ok / t.sum():.0%}"])
    write_table("r_eligibility", "Rabi forecast eligibility per season: potential forecasts, exclusions (missing "
                "data, never ``no crop'') and usable forecasts.", "tab:r_elig",
                ["Season", "Potential", "No recent image", "No target image", "Usable", "Share"], rows,
                "@{}lrrrrr@{}", size="\\footnotesize")

    # --- feature importance from the results file
    txt = (res / "FORECAST_RESULTS.md").read_text(encoding="utf-8")
    sec = txt.split("## Feature importance")[1]
    imp = re.findall(r"\|\s*([a-z_0-9]+)\s*\|\s*([0-9.]+)\s*\|", sec)
    nice = {"ndvi_15": "NDVI 15 days ago", "irr": "Irrigated flag", "clim_target": "Point normal at t+15",
            "ndvi_30": "NDVI 30 days ago", "season_min": "Season minimum NDVI", "ndvi_last": "Latest NDVI",
            "tmax_15": "Max temperature (15 d)", "doy_issue": "Day of season", "rain90": "Rain (90 d)",
            "wb60": "Water balance (60 d)", "chg30": "NDVI change (30 d)", "chg15": "NDVI change (15 d)",
            "season_max": "Season maximum NDVI", "clim_last": "Point normal at latest image", "rain30": "Rain (30 d)",
            "age_last": "Age of latest image", "et0_15": "ET0 (15 d)", "spei": "SPEI", "lat": "Latitude",
            "lon": "Longitude", "n_obs30": "Images in last 30 d", "rain_jun_ratio": "Rain since June vs normal"}
    wx = {"tmax_15", "rain90", "wb60", "rain30", "et0_15", "spei", "rain_jun_ratio"}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    names = [nice.get(k, k) for k, _ in imp][::-1]
    vals = [float(v) for _, v in imp][::-1]
    cols = [ORANGE if k in wx else NAVY for k, _ in imp][::-1]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.0%}", va="center", fontsize=7.5)
    ax.set_xlabel("Share of total gain (final XGBoost model)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NAVY, label="satellite / timing"), Patch(color=ORANGE, label="weather")],
              frameon=False, loc="lower right")
    savefig(fig, "res_r_importance")

    # --- predictions: error by month, scatter, examples
    p = pd.read_csv(res / "predictions.csv.gz", parse_dates=["issue_date"])
    p["month"] = p.issue_date.dt.month
    order = [10, 11, 12, 1, 2, 3, 4]
    mlab = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for col, nm, c, mk in (("pred_XGBoost", "XGBoost", NAVY, "o"), ("pred_persistence", "Persistence", GREY, "s"),
                           ("pred_anomaly", "Anomaly persistence", ORANGE, "^"),
                           ("pred_climatology", "Climatology", PURPLE, "v")):
        e = p.assign(e=(p.target - p[col]).abs()).groupby("month").e.mean().reindex(order)
        ax.plot(range(len(order)), e.values, marker=mk, color=c, label=nm, lw=1.6 if nm == "XGBoost" else 1.1)
    ax.set_xticks(range(len(order)), mlab)
    ax.set_ylabel("MAE of NDVI (test 2023-26)")
    ax.set_xlabel("Month of issue date")
    ax.legend(frameon=False, ncol=4, fontsize=7.5)
    savefig(fig, "res_r_month")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    actual = p.target - p.ndvi_last
    hexbins = []
    for ax, col, nm in ((axs[0], "pred_anomaly", "Anomaly persistence (best baseline)"),
                        (axs[1], "pred_XGBoost", "XGBoost")):
        pred = p[col] - p.ndvi_last
        hb = ax.hexbin(pred, actual, gridsize=60, extent=(-0.4, 0.4, -0.4, 0.4), bins="log", cmap="Blues", mincnt=1)
        hexbins.append(hb)
        ax.plot([-0.4, 0.4], [-0.4, 0.4], color=RED, lw=0.8)
        r = np.corrcoef(pred, actual)[0, 1]
        ax.set_title(f"{nm}\ncorrelation {r:.2f}", fontsize=9)
        ax.set_xlabel("Predicted NDVI change from latest image")
        ax.grid(False)
    axs[0].set_ylabel("Observed NDVI change from latest image")
    common_max = max(float(h.get_array().max()) for h in hexbins)
    for h in hexbins:
        h.set_clim(1, common_max)
    fig.colorbar(hb, ax=axs, shrink=0.8, label="forecasts (log)")
    savefig(fig, "res_r_scatter")

    obs = pd.read_csv(R / "data" / "gee_exports" / "C3_points_timeseries.csv", parse_dates=["date"])
    obs = obs[(obs["count"] >= 12) & obs["mean"].between(-0.2, 1)]
    q = p[(p.season == 2023) & p.established]
    cand = q.groupby("point_id").size()
    rng = np.random.default_rng(7)
    picks = []
    for mandal, irr in (("Adoni", 1), ("Kurnool", 0), ("Sira", 1), ("Pedda Kadubur", 0)):
        ids = [i for i in cand[cand >= 18].index if i.startswith(mandal.replace(" ", " ") + "_")]
        ids = [i for i in ids if q[q.point_id == i].irr.iloc[0] == irr]
        if ids:
            picks.append(rng.choice(ids))
    fig, axs = plt.subplots(2, 2, figsize=(11, 5.6))
    for ax, pid in zip(axs.flat, picks):
        o = obs[(obs.point_id == pid) & (obs.date >= "2023-10-01") & (obs.date <= "2024-04-30")].sort_values("date")
        o = o.sort_values("count").groupby("date").tail(1).sort_values("date")
        f = p[(p.point_id == pid) & (p.season == 2023)].copy()
        f["tdate"] = f.issue_date + pd.Timedelta(days=15)
        ax.plot(o.date, o["mean"], color="#1E1E1E", lw=1, marker=".", ms=4, label="Observed NDVI")
        ax.scatter(f.tdate, f.pred_XGBoost, s=16, color=NAVY, label="XGBoost forecast (+15 d)", zorder=4)
        ax.scatter(f.tdate, f.pred_persistence, s=12, color=GREY, marker="s", label="Persistence", zorder=3)
        irr = "irrigated" if f.irr.iloc[0] == 1 else "rainfed"
        ax.set_title(f"{pid.split('_')[0]} ({irr} point), rabi 2023-24", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_ylabel("NDVI")
    axs[0, 0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "res_r_examples")

    # --- C3 whole-mandal establishment (exploratory)
    c3 = pd.read_csv(R / "data" / "gee_exports" / "C3_mandal_season.csv")
    c3["share"] = 100 * c3.est_mar_ir / c3.obs_mar_ir
    piv = c3.pivot(index="mandal", columns="season", values="share")
    piv["normal"] = piv[[2019, 2020, 2021, 2022]].mean(axis=1)
    mand = ["Somandepalle", "Lepakshi", "Chilamathur", "Sira", "Agali", "Adoni", "Pedda Kadubur", "Kurnool", "Kallur"]
    rows = []
    for m in mand:
        r = piv.loc[m]
        cells = []
        for y in range(2019, 2026):
            v = r[y]
            s = f"{v:.0f}"
            if y >= 2023 and v < 0.75 * r.normal:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rows.append([m] + cells + [f"{r.normal:.0f}"])
    write_table("r_c3", "Exploratory: share (\\%) of irrigated cropland in each mandal that turned green (established) "
                "by March, per rabi season, from the whole-mandal Sentinel-2 summary. Bold = below 75\\,\\% of the "
                "2019--2022 normal. Seasons were known during design.", "tab:r_c3",
                ["Mandal", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25", "25-26", "Normal"], rows,
                "@{}lcccccccc@{}", size="\\footnotesize")

    # --- history of the rabi method
    rows = [["Kharif chain applied to rabi (replay)", "0 / 4", "0 / 10", "in-season images only"],
            ["Protocol v1 on 233 rabi-candidate fields", "0 / 4", "5 / 10", "development check 2023--26"],
            ["Early-drying signal (exploratory)", "2 / 4", "1 / 10", "post hoc, retrospective"],
            ["Whole-mandal 15-day NDVI forecast (final)", "\\multicolumn{2}{c}{forecasts greenness, not drought}",
             "", "18\\,\\% lower error, all test seasons"]]
    rows = [r if len(r) == 4 and "multicolumn" not in r[1] else [r[0], r[1], r[3]] for r in rows]
    lines = ["\\begin{table}[H]", "\\centering",
             "\\caption{Development of the rabi method: severe rabi mandal-seasons flagged and undeclared "
             "mandal-seasons flagged (2023-24 to 2025-26).}", "\\label{tab:r_history}", "\\footnotesize",
             "\\begin{tabular}{@{}lccl@{}}", "\\toprule",
             "\\textbf{Approach} & \\textbf{Severe flagged} & \\textbf{Undeclared flagged} & \\textbf{Note} \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TAB / "r_history.tex").write_text("\n".join(lines), encoding="utf-8")
    print("table r_history")


if __name__ == "__main__":
    weather()
    satellite()
    rabi()
