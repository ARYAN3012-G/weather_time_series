"""
Readable architecture diagrams for report v2 (main_v2.tex). Small canvas + large fonts so text stays
legible at full page width. Content verified against the module code (see main_v2.tex header).

    python final_report_materials/make_diagrams_v2.py
Writes latex/figures/v2_*.png (300 dpi, RGB).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent / "latex" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans"})
STYLE = {"data": ("#E6F0FA", "#2E6DB4"), "proc": ("#E9F5E9", "#2F7D32"), "model": ("#FFF3DE", "#C26A00"),
         "eval": ("#F1EAF8", "#6E4A9E"), "out": ("#FCE9E9", "#A83232"), "stage": ("#FFFFFF", "#1F3A5F"),
         "uav": ("#E3F4F4", "#1F7A7A")}
INK, NAVY = "#1E1E1E", "#1F3A5F"
TS, LS, LH = 11.5, 9.6, 2.15        # title size, line size (pt), line height (canvas units = 0.1 inch)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w / 10, h / 10))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), kind="proc", dashed=False, ts=TS, ls=LS, num=None):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fill, ec=edge,
                                lw=1.5, ls="--" if dashed else "-"))
    n = len(lines)
    block = 2.3 + (n * LH if n else 0)
    top = y + h / 2 + block / 2
    t = f"{num} {title}" if num else title
    ax.text(x + w / 2, top, t, ha="center", va="top", fontsize=ts, fontweight="bold", color=edge)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 2.5 - i * LH, line, ha="center", va="top", fontsize=ls, color=INK)
    return (x, y, w, h)


def arrow(ax, p1, p2, label=None, color=NAVY, ls="-", rad=0.0, lsize=8.8, loff=(0, 0.6), lha="center"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + loff[0], (p1[1] + p2[1]) / 2 + loff[1]
        ax.text(mx, my, label, ha=lha, va="bottom", fontsize=lsize, color=color, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.95))


def poly(ax, pts, color=NAVY, ls="-", head=True):
    """Orthogonal connector through pts; arrow head on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.4, ls=ls)
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, ls=ls)


def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def B(b): return (b[0] + b[2] / 2, b[1])
def L(b): return (b[0], b[1] + b[3] / 2)
def R(b): return (b[0] + b[2], b[1] + b[3] / 2)


def band(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=color + "14",
                                ec=color, lw=1.0, ls=(0, (4, 3))))
    ax.text(x + 1.2, y + h - 0.8, label, ha="left", va="top", fontsize=10.5, fontweight="bold", color=color)


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


# ------------------------------------------------------------------------------------------------
def system():
    fig, ax = canvas(100, 100)
    ax.text(25, 98.5, "KHARIF  (June – November)", ha="center", va="top", fontsize=12.5, fontweight="bold",
            color=NAVY)
    ax.text(75, 98.5, "RABI  (October – April)", ha="center", va="top", fontsize=12.5, fontweight="bold",
            color="#6E4A9E")
    x1, x2, w = 2, 52, 46
    k1 = box(ax, x1, 80, w, 13, "Data", ["AgERA5 daily weather, 1979–2026 (6 cells)",
                                         "Sentinel-2 10 m, 2019–2026 (240 fields)"], "data")
    k2 = box(ax, x1, 61, w, 14, "Module 1 · Weather forecast", ["hybrid physics + ML ensemble",
                                                                 "SPEI-30 drought class, 7 / 15 / 30 days"], "model")
    k3 = box(ax, x1, 41, w, 14, "Module 2 · Satellite confirmation", ["Sentinel-2 SVI on 240 random fields",
                                                                       "confirmed: median rainfed SVI < −1"], "model")
    k4 = box(ax, x1, 27, w, 9, "Stressed fields", ["list for drone survey"], "out")
    r1 = box(ax, x2, 80, w, 13, "Data", ["ERA5-Land root-zone soil, 1990–2023",
                                         "MODIS NDVI 2000–26 · Sentinel-2 points"], "data")
    r2 = box(ax, x2, 61, w, 14, "Season drought status", ["soil-moisture deficit duration (DD)",
                                                          "+ MODIS greenness, per district"], "model")
    r3 = box(ax, x2, 41, w, 14, "Field early-drying model", ["Random forest, 7 / 15 / 30 days ahead",
                                                             "warnings tagged soil-dry / soil-wet"], "model")
    r4 = box(ax, x2, 27, w, 9, "Fields to inspect", ["early-drying warnings"], "out")
    arrow(ax, B(k1), T(k2))
    arrow(ax, B(k2), T(k3), "alarm: where to look", loff=(1, -0.6), lha="left")
    arrow(ax, B(k3), T(k4))
    arrow(ax, B(r1), T(r2))
    arrow(ax, B(r2), T(r3), "soil percentile (soil check)", loff=(1, -0.6), lha="left")
    arrow(ax, B(r3), T(r4))
    u = box(ax, 12, 7, 76, 14, "Module 3 · UAV visual drought-stress assessment",
            ["drone survey → orthomosaic → 9-channel SegFormer-B2 (7 classes)",
             "class areas in m²  +  field-level visual stress flag"], "uav")
    arrow(ax, B(k4), (30, 21.2), "fields to fly", loff=(-1, -1.2), lha="right")
    arrow(ax, B(r4), (70, 21.2), "fields to fly", loff=(1, -1.2), lha="left")
    box(ax, 2, 0.6, 96, 4.2, "", [], "eval")
    ax.text(50, 2.7, "Independent checks, never used for training:  drought orders 2023–26  ·  district rabi yields "
                     "1997–2022  ·  UAV validation tiles", ha="center", va="center", fontsize=9.6,
            color=STYLE["eval"][1], fontweight="bold")
    save(fig, "v2_system")


def weather():
    fig, ax = canvas(100, 74)
    w = 22.5
    d = box(ax, 1, 58, w, 14, "AgERA5 weather", ["daily, 1979–2026", "6 independent cells", "rain, ET₀, T, RH …"], "data")
    p = box(ax, 26, 58, w, 14, "Water balance", ["D = P − ET₀", "30-day sum WB30"], "proc")
    s = box(ax, 51, 58, w, 14, "SPEI-30", ["log-logistic fit", "per site and month", "1979–2011, frozen"], "proc")
    lb = box(ax, 76, 58, w, 14, "Drought class", ["Healthy ≥ 0", "Moderate −1.5 … 0", "Severe < −1.5"], "proc")
    for a, b in [(d, p), (p, s), (s, lb)]:
        arrow(ax, R(a), L(b))
    f = box(ax, 1, 33, 31, 18, "Input features", ["30-day window t−29 … t", "9 weather variables",
                                                  "SPEI history, day of year", "+ water-balance probs"], "proc")
    wb = box(ax, 35, 33, 31, 18, "Water-balance ensemble", ["future SPEI window filled", "with each 1979–2011 year",
                                                            "→ class probabilities", "(physics baseline)"], "model")
    sp = box(ax, 69, 33, 30, 18, "Purged time split", ["train 1979–2011", "validation 2012–2018", "test 2019–2026",
                                                       "no sample crosses a split"], "eval")
    arrow(ax, B(d), T(f))
    arrow(ax, B(s), T(wb))
    arrow(ax, B(lb), T(sp), "target: class at t+h", loff=(0, -0.5))
    arrow(ax, L(wb), R(f))
    m = box(ax, 1, 9, 31, 18, "Models (5 seeds)", ["Random Forest, XGBoost", "LSTM, Transformer",
                                                  "soft-voting ensemble", "h = 7, 15, 30 days"], "model")
    bl = box(ax, 35, 9, 31, 18, "Baselines", ["persistence", "Markov, climatology", "water-balance ensemble"], "model")
    ev = box(ax, 69, 9, 30, 18, "Evaluation", ["macro-F1, balanced acc.", "severe POD / FAR", "bootstrap 95 % CI",
                                               "leakage tests"], "eval")
    arrow(ax, B(f), T(m))
    arrow(ax, B(sp), T(ev))
    arrow(ax, R(m), L(bl))
    arrow(ax, R(bl), L(ev))
    box(ax, 12, 0.5, 76, 5, "", [], "out")
    ax.text(50, 3, "Output: drought class per site, 7 / 15 / 30 days ahead → alarm for Module 2", ha="center",
            va="center", fontsize=10, fontweight="bold", color=STYLE["out"][1])
    arrow(ax, B(m), (25, 5.7))
    save(fig, "v2_weather")


def satellite():
    fig, ax = canvas(100, 78)
    band(ax, 0.5, 52, 99, 25, "Google Earth Engine", "#2E6DB4")
    s2 = box(ax, 2, 55, 23, 16, "Sentinel-2", ["2019–2026, 10 m", "cloud masks", "NDVI"], "data")
    fm = box(ax, 27.5, 55, 23, 16, "Farmland map", ["3 km around sites", "WorldCover cropland", "minus trees, built-up"],
             "proc")
    sg = box(ax, 53, 55, 21, 16, "Field segments", ["SNIC superpixels", "0.3–10 ha"], "proc")
    sm = box(ax, 76.5, 55, 21, 16, "240 random fields", ["20 rainfed + 10", "irrigated per site"], "proc")
    for a, b in [(s2, fm), (fm, sg), (sg, sm)]:
        arrow(ax, R(a), L(b))
    sv = box(ax, 52, 28, 46, 17, "Crop condition label (SVI)", ["SVI = (NDVI − normal) / spread",
                                                                 "per field and day of year",
                                                                 "normals ≤ 2023, scored year removed"], "proc")
    wl = box(ax, 2, 28, 46, 17, "Weather link (Module 1)", ["SPEI-30 / 60 / 90 history",
                                                            "weather-module forecasts"], "data")
    arrow(ax, B(sm), (sv[0] + 37, sv[1] + sv[3]))
    la = box(ax, 2, 6, 30, 16, "Lag analysis", ["SVI vs lagged SPEI", "rainfed: 10-day lag"], "eval")
    fc = box(ax, 35, 6, 30, 16, "Crop forecast", ["SVI class 15 / 30 d", "RF, XGBoost", "onset warning"], "model")
    rp = box(ax, 68, 6, 30, 16, "Operational replay", ["alarm: SPEI-30 < −1", "confirm: SVI < −1",
                                                       "→ fields_to_fly"], "out")
    poly(ax, [B(wl), (25, 25), (17, 25), (17, 22.3)], head=True)
    poly(ax, [B(sv), (75, 25), (83, 25), (83, 22.3)], head=True)
    arrow(ax, (50, 25), (50, 22.3))
    ax.plot([25, 75], [25, 25], color=NAVY, lw=1.4)
    ax.text(50, 1.5, "validated against AP / Karnataka drought orders 2023–2026 (comparison only)", ha="center",
            va="center", fontsize=9.6, color=STYLE["eval"][1], fontweight="bold")
    save(fig, "v2_satellite")


def rabi():
    fig, ax = canvas(100, 82)
    ax.text(14, 80.5, "Data", ha="center", va="top", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(50, 80.5, "Method", ha="center", va="top", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(85, 80.5, "Output", ha="center", va="top", fontsize=11.5, fontweight="bold", color=NAVY)
    rows = [(60, ("ERA5-Land root zone", ["45 points, 0.3° grid", "Sep–Apr, 1990–2023"]),
             ("Soil drought indicator", ["7-day mean → percentile vs 1990–2004",
                                         "deficit day: percentile ≤ 0.20",
                                         "DD = share of Nov–Feb days in deficit"]),
             ("Season drought status", ["per district", "warning by 15 Dec"])),
            (40, ("MODIS NDVI, 250 m", ["cropland, 3 districts", "16-day, 2000–2026"]),
             ("Greenness indicator", ["percentile vs 2000–2009", "VPCT = mean Dec–Feb percentile"]),
             ("Independent confirmation", ["13 later seasons", "2010–2022"])),
            (20, ("Sentinel-2 + AgERA5", ["1,350 random points", "2019–2026"]),
             ("Field early-drying model", ["Random forest, 39 causal features",
                                           "event: drying ≥ 21 d before normal",
                                           "warns 7 / 15 / 30 days ahead"]),
             ("Field warnings", ["tagged soil-dry / soil-wet", "→ inspection list → UAV"]))]
    meth = []
    for y, dd, mm, oo in rows:
        a = box(ax, 1, y, 26, 15, *dd, kind="data")
        b = box(ax, 32, y, 36, 15, *mm, kind="model")
        c = box(ax, 72, y, 24.5, 15, *oo, kind="out")
        arrow(ax, R(a), L(b))
        arrow(ax, R(b), L(c))
        meth.append(b)
    poly(ax, [(meth[0][0] + 6, meth[0][1]), (meth[0][0] + 6, 57.5), (30, 57.5), (30, 37.5),
              (30, 31), (meth[2][0], 31)], color="#6E4A9E", ls="--")
    ax.text(29.2, 46, "soil check", rotation=90, ha="center", va="center", fontsize=8.8, color="#6E4A9E",
            style="italic", bbox=dict(fc="white", ec="none", pad=0.4))
    v = box(ax, 1, 1, 98, 13, "Validation, never used for training",
            ["district rabi multi-crop yield index (APY 1997–2022), loss season = index ≤ −15 %",
             "soil DD: AUC 0.80 (by 15 Dec)  ·  MODIS VPCT: AUC 0.83  ·  SPEI: AUC 0.60"], "eval")
    for yy in (67.5, 47.5):
        ax.plot([96.6, 98.6], [yy, yy], color="#6E4A9E", lw=1.4, ls="--")
    ax.plot([98.6, 98.6], [67.5, 18], color="#6E4A9E", lw=1.4, ls="--")
    arrow(ax, (98.6, 18), (98.6, 14.4), color="#6E4A9E", ls="--")
    del v
    save(fig, "v2_rabi")


def uav_pipeline():
    fig, ax = canvas(100, 112)
    w, g = 22.6, 2.6
    xs = [1 + i * (w + g) for i in range(4)]
    band(ax, 0.3, 57, 99.4, 54, "A.  Building the training data and the model", "#C26A00")
    band(ax, 0.3, 0.5, 99.4, 50.5, "B.  Orthomosaic inference and field report", "#1F7A7A")
    h = 19
    r1 = [box(ax, xs[0], 85, w, h, "Drone photos", ["RGB images", "DJI FC8482", "camera"], "data", num="1."),
          box(ax, xs[1], 85, w, h, "Patch classifier", ["DINOv2 ViT-B/14", "multiscale patches", "test macro-F1",
                                                        "0.50–0.60"], "model", num="2."),
          box(ax, xs[2], 85, w, h, "Pseudo-labels", ["512 px window,", "stride 256, Hann", "dense CRF; conf.",
                                                     "< 0.55 → ignore"], "proc", num="3."),
          box(ax, xs[3], 85, w, h, "Annotation tool", ["custom (Tkinter,", "Matplotlib, OpenCV)", "brush correction",
                                                       "→ 279 clean tiles"], "proc", num="4.")]
    r2 = [box(ax, xs[3], 60, w, h, "Training mix", ["per epoch: 500 clean,", "500 pseudo, 1000", "synthetic composites",
                                                    "weights 8 / 0.5 / 1"], "proc", num="5."),
          box(ax, xs[2], 60, w, h, "9-channel input", ["RGB (ImageNet norm.)", "ExG, ExGR, CIVE,", "GLI, NGRDI, HSV-V"],
              "proc", num="6."),
          box(ax, xs[1], 60, w, h, "SegFormer-B2", ["focal + Dice + Lovász", "AdamW, 10 epochs", "best val. mIoU",
                                                    "(epoch 5)"], "model", num="7."),
          box(ax, xs[0], 60, w, h, "Logit adjustment", ["drought −1.2", "dry weed −0.4", "(chosen on", "validation)"],
              "model", num="8.")]
    r3 = [box(ax, xs[0], 29, w, h, "Orthomosaic", ["Agisoft Metashape", "3.8 Mpx – 2.86 Gpx", "GSD 1.7–18.5 mm"],
              "data", num="9."),
          box(ax, xs[1], 29, w, h, "Tiled prediction", ["1024 px tiles,", "stride 768", "9 channels → model"],
              "model", num="10."),
          box(ax, xs[2], 29, w, h, "Rolling stitching", ["softmax × Hann", "window, argmax;", "1 band in memory",
                                                         "(1.5 GB, not 80 GB)"], "proc", num="11."),
          box(ax, xs[3], 29, w, h, "Post-processing", ["no-data border,", "glare / shadow fix,", "small regions,",
                                                       "harvest zones"], "proc", num="12.")]
    r4 = [box(ax, xs[3], 4, w, h, "Area estimate", ["pixels × (GSD/1000)²", "→ m² per class"], "proc", num="13."),
          box(ax, xs[2], 4, w, h, "Stress decision", ["D > 0.15  or", "(N > 0.10 and", "C > 0.25)"], "eval", num="14."),
          box(ax, xs[1], 4, w, h, "Field report", ["mask PNG, overlay", "JPG, JSON, PDF", "per field"], "out", num="15.")]
    for a, b in zip(r1, r1[1:]):
        arrow(ax, R(a), L(b))
    arrow(ax, B(r1[3]), T(r2[0]))
    for a, b in zip(r2, r2[1:]):
        arrow(ax, L(a), R(b))
    for a, b in zip(r3, r3[1:]):
        arrow(ax, R(a), L(b))
    arrow(ax, B(r3[3]), T(r4[0]))
    for a, b in zip(r4, r4[1:]):
        arrow(ax, L(a), R(b))
    poly(ax, [B(r2[3]), (xs[0] + w / 2, 52.0), (xs[1] + w / 2, 52.0), T(r3[1])], color="#C26A00")
    ax.text(xs[0] + w + 1.3, 53.2, "trained model", fontsize=9, color="#C26A00", style="italic", ha="center",
            va="bottom", bbox=dict(fc="white", ec="none", pad=0.4))
    arrow(ax, R(r3[0]), L(r3[1]))
    save(fig, "v2_uav_pipeline")


def uav_model():
    fig, ax = canvas(100, 66)
    inp = box(ax, 0.5, 44, 14, 17, "Input", ["9 × 512 × 512", "(1024 at", "inference)"], "data")
    pe = box(ax, 17, 44, 14, 17, "Patch embed", ["7×7 conv,", "stride 4", "9 → 64 ch"], "proc")
    st = []
    specs = [("Stage 1", "64 ch, ×3", "H/4", "R = 8"), ("Stage 2", "128 ch, ×4", "H/8", "R = 4"),
             ("Stage 3", "320 ch, ×6", "H/16", "R = 2"), ("Stage 4", "512 ch, ×3", "H/32", "R = 1")]
    for i, (t, a, b, c) in enumerate(specs):
        st.append(box(ax, 33.5 + i * 16.6, 44, 15, 17, t, [a, b, c], "model"))
    arrow(ax, R(inp), L(pe))
    arrow(ax, R(pe), L(st[0]))
    for a, b in zip(st, st[1:]):
        arrow(ax, R(a), L(b))
    ax.text(66.5, 63.8, "Mix-Transformer encoder (MiT-B2): efficient self-attention (reduction R) + Mix-FFN",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color=STYLE["model"][1])
    mlp = []
    for i in range(4):
        mlp.append(box(ax, 33.5 + i * 16.6, 26, 15, 11, "MLP", [f"{[64, 128, 320, 512][i]} → 768", "up to H/4"],
                       "proc", ts=10.5, ls=9))
        arrow(ax, B(st[i]), T(mlp[i]))
    fuse = box(ax, 33.5, 5, 32, 14, "Concatenate + fuse", ["4 × 768 → 768 (1×1 conv)", "7-class head (1×1 conv)"],
               "proc")
    up = box(ax, 68.5, 5, 31, 14, "Output", ["bilinear ×4 → 7 × H × W", "logit adjustment, argmax"], "out")
    for m in mlp:
        poly(ax, [B(m), (B(m)[0], 22.5), (49.5, 22.5), (49.5, 19.2)], head=False)
    arrow(ax, (49.5, 22.5), (49.5, 19.3))
    arrow(ax, R(fuse), L(up))
    box(ax, 0.5, 15, 30, 22, "Weight widening (3 → 9)", ["pretrained RGB conv:", "W[:, 0:3] = W_old / 3", "W[:, 3:9] = 0 (new channels",
                                                "start at zero and learn)"],
        "stage", ts=10.5, ls=9.2)
    arrow(ax, (15.5, 38.2), B(pe), color="#8A8A8A", ls="--")
    save(fig, "v2_uav_model")


if __name__ == "__main__":
    system()
    weather()
    satellite()
    rabi()
    uav_pipeline()
    uav_model()
