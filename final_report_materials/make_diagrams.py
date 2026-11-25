"""
Architecture and flow diagrams for the final-semester report (weather + satellite + rabi modules).

    python final_report_materials/make_diagrams.py
Writes final_report_materials/diagrams/<name>.png (300 dpi) and .svg
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import os  # noqa: E402

NO_TITLE = bool(os.environ.get("NO_TITLE"))   # NO_TITLE=1 -> versions without the drawn title (for LaTeX captions)
OUT = Path(__file__).resolve().parent / ("latex/figures" if NO_TITLE else "diagrams")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

STYLE = {  # fill, edge
    "data": ("#E6F0FA", "#2E6DB4"),
    "proc": ("#E9F5E9", "#2F7D32"),
    "model": ("#FFF3DE", "#C26A00"),
    "eval": ("#F1EAF8", "#6E4A9E"),
    "out": ("#FCE9E9", "#A83232"),
    "stage": ("#FFFFFF", "#1F3A5F"),
    "plan": ("#FAFAFA", "#8A8A8A"),
}
INK, GREY = "#1E1E1E", "#555555"


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    if not NO_TITLE:
        ax.text(w * 5, h * 10 - 3, title, ha="center", va="top", fontsize=15, fontweight="bold", color="#1F3A5F")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), kind="proc", dashed=False, tsize=10.5, lsize=8.6):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=edge,
                                lw=1.6, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=edge)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.6 - i * 3.6, l, ha="center", va="top", fontsize=lsize, color=INK)
    return (x, y, w, h)


def arrow(ax, p1, p2, label=None, color="#1F3A5F", style="-|>", ls="-", rad=0.0, lsize=8, loff=(0, 1.2)):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, lw=1.4, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + loff[0], (p1[1] + p2[1]) / 2 + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=lsize, color=color, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bot(b):
    return (b[0] + b[2] / 2, b[1])


def left(b):
    return (b[0], b[1] + b[3] / 2)


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def legend(ax, x, y, kinds):
    names = {"data": "Data source", "proc": "Processing", "model": "Model", "eval": "Evaluation / validation",
             "out": "Output", "plan": "Planned / external"}
    for i, k in enumerate(kinds):
        fill, edge = STYLE[k]
        ax.add_patch(FancyBboxPatch((x + i * 26, y), 4, 2.6, boxstyle="round,pad=0.2", fc=fill, ec=edge, lw=1.2,
                                    ls="--" if k == "plan" else "-"))
        ax.text(x + i * 26 + 5.5, y + 1.3, names[k], va="center", fontsize=8.5, color=GREY)



def _rgb(path):
    """Drop the (unused) alpha channel: identical pixels, much faster for pdfLaTeX."""
    from PIL import Image
    im = Image.open(path)
    if im.mode == "RGBA":
        im.convert("RGB").save(path, dpi=im.info.get("dpi", (300, 300)))

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    _rgb(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------
def fig1_system():
    fig, ax = canvas(17, 10.6, "Figure 1. Overall system architecture: weather → satellite → UAV")
    W = 170
    # data sources
    ds = [("AgERA5 v2.0 weather", ["Copernicus CDS, daily", "6 independent 0.1° cells", "1979 – Nov 2026"]),
          ("Sentinel-2 SR", ["Google Earth Engine", "10 m, every ~5 days", "Cloud Score+ / SCL masks"]),
          ("Land cover", ["ESA WorldCover 2020/21", "Dynamic World", "Mandal polygons (2018)"]),
          ("UAV imagery", ["DeepDroughtUAV team", "10 surveyed fields", "2,254 images"]),
          ("Drought orders (G.O.s)", ["AP G.O.Ms 2023–2026", "Karnataka 2023", "kharif + rabi"])]
    bw, gap = 29.5, 4.6
    srcs = []
    for i, (t, l) in enumerate(ds):
        srcs.append(box(ax, 4 + i * (bw + gap), 76, bw, 17, t, l, "data", dashed=(i == 3)))
    # stages
    s1 = box(ax, 4, 41, 38, 24, "STAGE 1 · WEATHER: PREDICT",
             ["SPEI-30 drought class", "7 / 15 / 30 days ahead", "physics + ML hybrid (XGBoost)",
              "water-balance ensemble baseline"], "stage")
    s2 = box(ax, 50, 41, 38, 24, "STAGE 2 · SATELLITE: CONFIRM",
             ["240 random fields (kharif)", "Standardized Veg. Index (SVI)", "lag analysis, RF / XGBoost",
              "operational replay"], "stage")
    s2b = box(ax, 96, 41, 34, 24, "STAGE 2b · RABI MODULE",
              ["NDVI + early-drying experiments", "ERA5-Land soil input (5 cells)", "7 / 15 / 30-day soil-risk models",
               "regional risk, not field diagnosis"], "stage")
    s3 = box(ax, 138, 41, 28, 24, "STAGE 3 · UAV: QUANTIFY", ["drone survey of", "flagged fields", "segmentation of",
                                                              "damaged area"], "plan", dashed=True)
    # outputs
    outs = [box(ax, 4, 12, 38, 17, "Drought alarm", ["per site, forecast class", "t+7 / t+15 / t+30"], "out"),
            box(ax, 50, 12, 38, 17, "Stressed-field list", ["SYSTEM_REPORT per date", "fields_to_fly.kml"], "out"),
            box(ax, 96, 12, 34, 17, "Rabi risk + crop context", ["regional soil-deficit scores", "available satellite context",
                                                                "inspection priorities"], "out"),
            box(ax, 138, 12, 28, 17, "Damaged area", ["(UAV module,", "added later)"], "plan", dashed=True)]
    # arrows: sources -> stages
    arrow(ax, bot(srcs[0]), top(s1))
    arrow(ax, bot(srcs[0]), (s2[0] + 8, s2[1] + s2[3]), "SPEI link", rad=0.0, loff=(-6, 0))
    arrow(ax, bot(srcs[1]), (s2[0] + 26, s2[1] + s2[3]))
    arrow(ax, bot(srcs[1]), (s2b[0] + 8, s2b[1] + s2b[3]))
    arrow(ax, bot(srcs[2]), (s2b[0] + 24, s2b[1] + s2b[3]))
    arrow(ax, bot(srcs[2]), (s2[0] + 34, s2[1] + s2[3]), rad=0.1)
    arrow(ax, bot(srcs[3]), top(s3), ls="--", color="#8A8A8A")
    # stage chain
    arrow(ax, right(s1), left(s2), "where to look")
    arrow(ax, (s2[0] + s2[2] - 2, s2[1]), (s3[0] + 2, s3[1]), "which fields to fly", rad=0.3, loff=(-13, -11.5))
    for s, o in zip([s1, s2, s2b, s3], outs):
        arrow(ax, bot(s), top(o), ls="--" if s is s3 else "-", color="#8A8A8A" if s is s3 else "#1F3A5F")
    # validation band
    v = box(ax, 4, 1, 162, 7.5, "", [], "eval")
    ax.text(85, 4.8, "Independent validation: AP / Karnataka drought orders (comparison, not tuning) · UAV field "
                     "verdicts · held-out years · leakage tests · baselines", ha="center", va="center",
            fontsize=9.5, color=STYLE["eval"][1], fontweight="bold")
    arrow(ax, (168.5, 76), (168.5, 8.6), ls=":", color=STYLE["eval"][1])
    ax.text(169.5, 50, "compare", fontsize=8, color=STYLE["eval"][1], style="italic", rotation=90, va="center")
    legend(ax, 30, 96, ["data", "out", "eval", "plan"])
    del W
    save(fig, "fig1_system_architecture")


def fig2_weather():
    fig, ax = canvas(17, 11, "Figure 2. Weather module (Stage 1) architecture")
    d1 = box(ax, 4, 82, 36, 17, "AgERA5 v2.0 (CDS)", ["daily, 1979-01-30 – 2026-11-02", "6 independent 0.1° cells",
                                                       "rain, ET0 (FAO-56), T, RH, wind, rad"], "data")
    p1 = box(ax, 48, 82, 36, 17, "Data preparation", ["site audit: duplicate cells removed", "water balance = P – ET0",
                                                      "WB30 accumulation"], "proc")
    p2 = box(ax, 92, 82, 36, 17, "SPEI-30 calibration", ["log-logistic, L-moments", "per site × calendar month",
                                                         "fitted 1979–2011 only, frozen"], "proc")
    p3 = box(ax, 136, 82, 32, 17, "Drought labels", ["Healthy  SPEI ≥ 0", "Moderate  –1.5 … 0",
                                                    "Severe  < –1.5"], "proc")
    arrow(ax, right(d1), left(p1))
    arrow(ax, right(p1), left(p2))
    arrow(ax, right(p2), left(p3))
    f = box(ax, 4, 49, 52, 25, "Feature sets (30-day input window t-29 … t)",
            ["weather: 9 AgERA5 variables", "weather_spei: + SPEI history (≤ t)", "spei_only: SPEI history",
             "weather_spei_doy: + sin/cos day of year", "weather_spei_doy_wb (hybrid): + 3 WB-ensemble probs"], "proc")
    wb = box(ax, 64, 49, 40, 25, "Water-balance ensemble (physics)",
             ["observed part of the SPEI-30 window", "+ each 1979–2011 year's water", "balance for the unknown days",
              "→ class probabilities (baseline", "and hybrid input)"], "model")
    sp = box(ax, 112, 49, 56, 25, "Purged chronological split",
             ["TRAIN 1979–2011 · VAL 2012–2018 · TEST 2019–2026", "sample kept only if input…target span is",
              "inside one period", "LOCO: train one cluster, test the other", "(Kurnool vs south)"], "eval")
    arrow(ax, bot(p3), (sp[0] + sp[2] / 2, sp[1] + sp[3]), "target: class at t + h")
    arrow(ax, bot(d1), top(f))
    arrow(ax, bot(p2), top(wb))
    arrow(ax, left(wb), right(f), "hybrid input")
    m = box(ax, 4, 20, 64, 24, "Models (5 seeds each)",
            ["Random Forest · XGBoost (flattened 30 × F input)", "LSTM · Transformer (sequence input)",
             "soft-voting ensemble weighted by validation macro-F1", "early stopping, scalers, model choice: TRAIN/VAL only",
             "horizons h = 7, 15, 30 days"], "model")
    b = box(ax, 76, 20, 38, 24, "Baselines", ["persistence (class today)", "Markov transition (train)",
                                             "climatology (calendar day)", "majority class",
                                             "water-balance ensemble"], "model")
    e = box(ax, 122, 20, 46, 24, "Evaluation (TEST touched once)",
            ["macro-F1, balanced accuracy, ROC/PR-AUC", "Brier skill, severe POD / FAR / CSI",
             "block-bootstrap 95% CI (target month)", "LOCO + Marusov et al. protocol", "22 leakage / integrity tests"],
            "eval")
    arrow(ax, bot(f), (m[0] + m[2] / 2, m[1] + m[3]))
    arrow(ax, bot(wb), (b[0] + b[2] / 2, b[1] + b[3]))
    arrow(ax, bot(sp), (e[0] + e[2] / 2, e[1] + e[3]))
    arrow(ax, right(m), left(b))
    arrow(ax, right(b), left(e))
    o = box(ax, 40, 2, 92, 12, "Output: drought class forecast per site (t+7 / t+15 / t+30)",
            ["Hybrid ensemble macro-F1 0.819 / 0.651 / 0.424 vs persistence 0.662 / 0.515 / 0.424 (test 2019–2026)"],
            "out", lsize=9)
    arrow(ax, bot(m), (o[0] + 10, o[1] + o[3]))
    arrow(ax, bot(e), (o[0] + o[2] - 10, o[1] + o[3]))
    legend(ax, 4, 3.5, [])
    save(fig, "fig2_weather_module_architecture")


def bus(ax, sources, targets, ybus, color="#1F3A5F"):
    """Orthogonal connector: each source drops to ybus, one horizontal bus, arrows down to each target."""
    xs = [p[0] for p in sources] + [p[0] for p in targets]
    for x, y in sources:
        ax.plot([x, x], [y, ybus], color=color, lw=1.4, solid_capstyle="round")
    ax.plot([min(xs), max(xs)], [ybus, ybus], color=color, lw=1.4, solid_capstyle="round")
    for x, y in targets:
        arrow(ax, (x, ybus), (x, y), color=color)


def fig3_satellite():
    fig, ax = canvas(17, 13, "Figure 3. Satellite module (Stage 2, kharif) architecture")
    ax.add_patch(FancyBboxPatch((2, 92), 166, 29, boxstyle="round,pad=0.4,rounding_size=2", fc="#F7FAFD",
                                ec="#9DB8D9", lw=1.2, ls="--"))
    ax.text(5, 118.8, "Google Earth Engine", fontsize=10, color="#2E6DB4", fontweight="bold", va="top")
    a = box(ax, 5, 104, 36, 12.5, "Sentinel-2 SR Harmonized", ["2019 – Oct 2026 · CS+ ≥ 0.6 + SCL",
                                                               "B2–B12, NDVI, NDMI, EVI"], "data", lsize=8.2)
    lc = box(ax, 5, 93.5, 36, 8.5, "WorldCover + Dynamic World", ["cropland minus built-up / trees / water"], "data",
             lsize=8.0, tsize=9.5)
    c = box(ax, 48, 95, 36, 19, "Script A: farmland", ["3 km around 8 sites", "crop cycle in ≥ 2 years",
                                                       "irrigated: green Mar–May ≥ 3 yrs"], "proc")
    s = box(ax, 91, 95, 34, 19, "SNIC segmentation", ["13,750 field segments", "0.3 – 10 ha", "seeded random sample"],
            "proc")
    t = box(ax, 132, 95, 33, 19, "Script B: time series", ["Sentinel-2 series of", "240 fields (20 rainfed",
                                                          "+ 10 irrigated per site)"], "proc")
    arrow(ax, right(a), (c[0], a[1] + a[3] / 2))
    arrow(ax, right(lc), (c[0], lc[1] + lc[3] / 2))
    arrow(ax, right(c), left(s))
    arrow(ax, right(s), left(t))
    # row 2 (right to left)
    cl = box(ax, 118, 62, 47, 22, "Cleaning (Python)", ["≥ 60% clear pixels per field", "same-day duplicates merged",
                                                        "5-day grid: causal (inputs) /", "interp (labels) / reference"],
             "proc")
    sv = box(ax, 62, 62, 50, 22, "Labels: Standardized Veg. Index",
             ["SVI = (NDVI – normal) / spread", "per field and day of year", "normals: images ≤ 31 Dec 2023,",
              "scored year removed"], "proc")
    wl = box(ax, 5, 62, 50, 22, "Weather link (from Stage 1)", ["SPEI-30/60/90 on the 5-day grid",
                                                                "frozen 1979–2011 calibration",
                                                                "weather-module forecasts t+7/15/30"], "data")
    arrow(ax, bot(t), (t[0] + t[2] / 2, cl[1] + cl[3]))
    arrow(ax, left(cl), right(sv))
    # row 3
    an = box(ax, 5, 24, 50, 26, "Lag analysis", ["VCI / SVI vs SPEI-k(t – lag)", "lags 0–90 days",
                                                 "alarm confirmation within 60 d", "rainfed: SPEI-30, 10-day lag"],
             "eval")
    fm = box(ax, 62, 24, 50, 26, "Crop-condition forecast", ["SVI class 15 / 30 days ahead",
                                                             "RF + XGBoost, 5 seeds, 4 versions:",
                                                             "1 satellite · 2 + past weather",
                                                             "3 + weather forecast · 4 + future wx",
                                                             "TRAIN 2019–23 · VAL 2024 · TEST 2025–26"], "model")
    rp = box(ax, 118, 24, 47, 26, "Operational replay (no ML)", ["issue every 5 days, Jun – Nov",
                                                                 "weather 7 d late, satellite 2 d late",
                                                                 "alarm: SPEI-30 < –1 or severe fc.",
                                                                 "confirm: median rainfed SVI < –1",
                                                                 "from ≥ 5 clear fields"], "model")
    bus(ax, [bot(wl), bot(sv)], [top(an), top(fm), top(rp)], 56)
    o1 = box(ax, 5, 2, 78, 14, "Outputs", ["SYSTEM_REPORT per date · fields_to_fly.csv / .kml",
                                           "(red = damaged, orange = at risk in 15 days) → Stage 3 UAV"], "out")
    o2 = box(ax, 90, 2, 75, 14, "Validation", ["AP / Karnataka orders 2023–26: severe 9/9 confirmed,",
                                               "5/9 before the order · false alarms 1/11 · 18 leakage tests"],
             "eval")
    bus(ax, [bot(an), bot(fm), bot(rp)], [top(o1), top(o2)], 19.5)
    save(fig, "fig3_satellite_module_architecture")


def fig4_rabi():
    fig, ax = canvas(17, 14.5, "Figure 4. Rabi module architecture (15-day vegetation forecast)")
    ax.add_patch(FancyBboxPatch((2, 107), 166, 29, boxstyle="round,pad=0.4,rounding_size=2", fc="#F7FAFD",
                                ec="#9DB8D9", lw=1.2, ls="--"))
    ax.text(5, 133.8, "Google Earth Engine · script C3", fontsize=10, color="#2E6DB4", fontweight="bold", va="top")
    m = box(ax, 5, 108, 36, 22, "Mandal boundaries", ["9 mandals (LGD 2018 polygons)", "Somandepalle, Lepakshi,",
                                                      "Chilamathur, Sira, Agali, Adoni,",
                                                      "Pedda Kadubur, Kurnool, Kallur"], "data")
    cr = box(ax, 48, 108, 36, 22, "Cropland + irrigation", ["WorldCover 2020 AND 2021", "(fixed denominator)",
                                                           "irrigated: Mar–May NDVI p75", "> 0.4 in ≥ 2 of 2019–23"],
             "proc")
    sp = box(ax, 91, 108, 34, 22, "Stratified random points", ["per mandal: 100 rainfed", "+ 50 irrigated (seed 42)",
                                                              "= 1,350 points"], "proc")
    ts = box(ax, 132, 108, 33, 22, "Point time series", ["Sentinel-2 SR, CS+ + SCL", "15 m buffer mean NDVI",
                                                        "Oct–Apr, 2019-20 – 2025-26", "355,043 rows"], "data")
    for p, q in ((m, cr), (cr, sp), (sp, ts)):
        arrow(ax, right(p), left(q))
    # row 2 (right to left)
    cl = box(ax, 128.5, 73, 33, 24, "Cleaning", ["≥ 12 of ~16 pixels clear", "NDVI in [–0.2, 1]",
                                               "one obs per point-date", "313,797 clean obs"], "proc")
    ins = box(ax, 62, 73, 56, 24, "Forecast instances (every 5 days)",
              ["inputs: images ≤ t – 2 d, latest ≤ 12 d old", "target: real image at t + 15 ± 3 d",
               "(no interpolation); excluded = missing data", "233,020 of 342,900 usable (68%)"], "proc")
    ft = box(ax, 5, 73, 50, 24, "Features", ["NDVI now, –15 d, –30 d, changes, ages",
                                             "season min / max, obs count, DOY",
                                             "point normal at t + 15 (training seasons)",
                                             "AgERA5 weather 8 d late, frozen SPEI"], "proc")
    arrow(ax, bot(ts), (ts[0] + ts[2] / 2, cl[1] + cl[3]))
    arrow(ax, left(cl), right(ins))
    arrow(ax, left(ins), right(ft))
    # row 3
    spl = box(ax, 5, 38, 50, 24, "Chronological split", ["TRAIN 2019-20 … 2021-22", "VAL 2022-23 (early stopping)",
                                                         "refit on 2019-20 … 2022-23",
                                                         "TEST 2023-24, 2024-25, 2025-26"], "eval")
    mo = box(ax, 62, 38, 56, 24, "XGBoost regression", ["target: NDVI change to future image", "tree count chosen on validation",
                                                        "+ satellite-only version",
                                                        "+ 1-day-weather sensitivity"], "model")
    bl = box(ax, 125, 38, 40, 24, "Baselines", ["persistence (NDVI now)", "climatology (point normal)",
                                               "anomaly persistence"], "model")
    arrow(ax, bot(ft), top(spl))
    arrow(ax, right(spl), left(mo))
    arrow(ax, right(mo), left(bl), "same test rows", loff=(0, 1.0))
    ev = box(ax, 5, 2, 96, 22, "Evaluation (rules registered in git before training)",
             ["MAE and skill per test season, rainfed vs irrigated, Dec–Mar subset",
              "bootstrap over sampled points and over weather cells",
              "decline warnings: AUC, precision, recall (drought or harvest)"], "eval")
    rs = box(ax, 108, 2, 57, 22, "Result (historical evaluation)",
             ["MAE 0.051 vs 0.062 (best baseline)", "18% lower error overall",
              "beats all baselines in each test season", "weather features did not lower MAE"], "out")
    bus(ax, [bot(mo), bot(bl)], [(mo[0] + mo[2] / 2, ev[1] + ev[3]), top(rs)], 30)
    save(fig, "fig4_rabi_module_architecture")


def fig5_timeline():
    """Operational data flow on one issue date t (delays and horizons)."""
    fig, ax = canvas(16, 6.5, "Figure 5. What is known on forecast date t (operational timing)")
    x0, x1 = 10, 150

    def X(day):  # day relative to t, -40 .. +32
        return x0 + (day + 40) / 72 * (x1 - x0)
    y = 30
    ax.plot([X(-40), X(32)], [y, y], color=INK, lw=1.4)
    for dd in range(-40, 33, 5):
        ax.plot([X(dd), X(dd)], [y - 0.8, y + 0.8], color=INK, lw=0.8)
        ax.text(X(dd), y - 2.5, f"{dd:+d}" if dd else "t", ha="center", va="top", fontsize=8, color=GREY)
    ax.axvline(X(0), ymin=0.38, ymax=0.83, color="#A83232", lw=1.6, ls="--")
    ax.text(X(0), 56, "forecast issued (t)", ha="center", fontsize=9.5, color="#A83232", fontweight="bold")
    lanes = [(47, "AgERA5 weather", -40, -8, "#2E6DB4", "available up to t – 8 days (publication delay)"),
             (41, "Sentinel-2 image", -40, -2, "#2F7D32", "available up to t – 2 days"),
             (35, "SPEI-30 window", -38, -8, "#6E4A9E", "30 days of water balance")]
    for yy, name, a, b, col, note in lanes:
        ax.add_patch(FancyBboxPatch((X(a), yy - 1.6), X(b) - X(a), 3.2, boxstyle="round,pad=0.1", fc=col, ec=col,
                                    alpha=0.75))
        ax.text(X(a) - 1, yy, name, ha="right", va="center", fontsize=9, color=col, fontweight="bold")
        ax.text(X(a) + 2, yy, note, ha="left", va="center", fontsize=8.5, color="white", fontweight="bold")
    tg = [(7, "t+7: weather class", "center"),
          (15, "t+15: weather class · crop SVI · rabi NDVI (target image within ± 3 d)", "center"),
          (30, "t+30: weather class · crop SVI", "right")]
    for i, (h, lab, ha) in enumerate(tg):
        ax.annotate("", xy=(X(h), y + 1), xytext=(X(0), y + 1),
                    arrowprops=dict(arrowstyle="-|>", color="#C26A00", lw=1.3,
                                    connectionstyle=f"arc3,rad={-0.25 - 0.1 * i}"))
        ly = 20 - i * 4.5
        ax.plot([X(h), X(h)], [y - 4, ly + 2.2], color="#C26A00", lw=0.7, ls=":")
        ax.text(X(h) + (1 if ha == "right" else 0), ly, lab, ha=ha, va="bottom", fontsize=8.5, color="#C26A00",
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(X(-40), 4, "Rabi NDVI horizons start at issue t; input delay does not subtract from them.\n"
                       "The regional soil-deficit model has a separate architecture and reference period.", fontsize=8.8,
            color=GREY)
    save(fig, "fig5_operational_timing")


def fig6_splits():
    fig, ax = canvas(16, 6, "Figure 6. Training, validation and test periods of the three modules")
    rows = [("Weather (daily, SPEI)", [(1979, 2012, "TRAIN 1979–2011", "#8FB3D9"), (2012, 2019, "VAL 2012–18", "#F2C14E"),
                                       (2019, 2026.9, "TEST 2019–26", "#C0504D")], 1979),
            ("Satellite, kharif (SVI)", [(2019, 2024, "TRAIN 2019–23", "#8FB3D9"), (2024, 2025, "VAL 2024", "#F2C14E"),
                                         (2025, 2026.9, "TEST 2025–26", "#C0504D")], 2019),
            ("Rabi (NDVI, seasons Oct–Apr)", [(2019.75, 2022.75, "TRAIN 19-20 … 21-22", "#8FB3D9"),
                                              (2022.75, 2023.75, "VAL 22-23", "#F2C14E"),
                                              (2023.75, 2026.33, "TEST 23-24 … 25-26", "#C0504D")], 2019)]
    # two panels: long weather bar (1979-2027) and zoom (2019-2027)
    def X(yr, lo, hi, a=40, b=155):
        return a + (yr - lo) / (hi - lo) * (b - a)
    ys = [42, 27, 14]
    for (name, segs, lo), y in zip(rows, ys):
        lo_, hi_ = (1979, 2027) if lo == 1979 else (2019, 2027)
        ax.text(38, y + 2, name, ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK)
        for a, b, lab, col in segs:
            ax.add_patch(FancyBboxPatch((X(a, lo_, hi_), y), X(b, lo_, hi_) - X(a, lo_, hi_), 4,
                                        boxstyle="round,pad=0.05", fc=col, ec="white", lw=1.5))
            ax.text((X(a, lo_, hi_) + X(b, lo_, hi_)) / 2, y + 2, lab, ha="center", va="center", fontsize=8.3,
                    color="white" if col != "#F2C14E" else INK, fontweight="bold")
        ticks = range(1979, 2028, 8) if lo == 1979 else range(2019, 2028)
        for t in ticks:
            ax.text(X(t, lo_, hi_), y - 1.2, str(t), ha="center", va="top", fontsize=7.5, color=GREY)
    ax.text(80, 3, "Normals and calibrations use training periods only (SPEI 1979–2011, SVI 2019–2023, rabi point "
                   "normals from training seasons). Government orders are never used for training.",
            ha="center", fontsize=8.6, color=GREY)
    save(fig, "fig6_data_splits")


def fig7_rabi_logic():
    fig, ax = canvas(16, 7.5, "Figure 7. One rabi forecast: eligibility and target rules")
    a = box(ax, 4, 42, 30, 20, "Issue date t", ["every 5 days", "15 Oct … t + 18 ≤ 30 Apr"], "data")
    b = box(ax, 44, 42, 34, 20, "Recent image?", ["latest image ≤ t – 2 d", "and ≤ 12 days old"], "proc")
    c = box(ax, 88, 42, 34, 20, "Real target?", ["image within", "t + 12 … t + 18 days"], "proc")
    d = box(ax, 132, 42, 26, 20, "Usable", ["features → XGBoost", "and baselines"], "out")
    arrow(ax, right(a), left(b))
    arrow(ax, right(b), left(c), "yes")
    arrow(ax, right(c), left(d), "yes")
    e = box(ax, 44, 8, 34, 18, "Excluded: no recent obs", ["54,260 forecasts", "counted as missing data,",
                                                          "never as 'no crop'"], "eval")
    f = box(ax, 88, 8, 34, 18, "Excluded: no target", ["55,620 forecasts", "no interpolation", "across gaps"], "eval")
    arrow(ax, bot(b), top(e), "no", loff=(3, 0))
    arrow(ax, bot(c), top(f), "no", loff=(3, 0))
    ax.text(145, 30, "233,020\nusable (68%)", ha="center", fontsize=11, color=STYLE["out"][1], fontweight="bold")
    save(fig, "fig7_rabi_forecast_rules")


if __name__ == "__main__":
    fig1_system()
    fig2_weather()
    fig3_satellite()
    fig4_rabi()
    fig5_timeline()
    fig6_splits()
    fig7_rabi_logic()
