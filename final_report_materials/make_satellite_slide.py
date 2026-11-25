"""
Wide, large-font version of the satellite-module diagram for the presentation (one full slide).
Same content as v2_satellite in make_diagrams_v2.py, fewer words per box so it stays readable when projected.

    python final_report_materials/make_satellite_slide.py
Writes latex/figures/v2_satellite_slide.png (300 dpi, RGB).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from make_system_slide import NAVY, OUT, STYLE, arrow, box  # noqa: E402

STYLE.update({"data": ("#E6F0FA", "#2E6DB4"), "proc": ("#E9F5E9", "#2F7D32")})
W, H = 80, 40


def main():
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    x0, gap = 0.6, 2.4
    bw = (W - 2 * x0 - 3 * gap) / 4
    xs = [x0 + i * (bw + gap) for i in range(4)]
    r1, r2, r3, bh = 26.4, 14.4, 3.4, 8.6

    ax.add_patch(FancyBboxPatch((0.3, r1 - 0.9), W - 0.6, bh + 4.4, boxstyle="round,pad=0.1,rounding_size=1.0",
                                fc="#2E6DB414", ec="#2E6DB4", lw=1.0, ls=(0, (4, 3))))
    ax.text(1.0, r1 + bh + 3.0, "Google Earth Engine", ha="left", va="top", fontsize=10, fontweight="bold",
            color="#2E6DB4")

    a1 = box(ax, xs[0], r1, bw, bh, "Sentinel-2", ["2019–2026, 10 m", "cloud-masked NDVI"], "data")
    a2 = box(ax, xs[1], r1, bw, bh, "Farmland map", ["WorldCover cropland", "Dynamic World removes", "built-up, trees, water"], "proc")
    a3 = box(ax, xs[2], r1, bw, bh, "Field segments", ["crop cycle ≥ 2 of 7 yrs", "edges removed", "SNIC, 0.3–10 ha"], "proc")
    a4 = box(ax, xs[3], r1, bw, bh, "240 random fields", ["20 rainfed + 10", "irrigated per site"], "proc")
    for a, b in ((a1, a2), (a2, a3), (a3, a4)):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2))

    ww = 2 * bw + gap
    wl = box(ax, xs[0], r2, ww, bh, "Weather link (Module 1)", ["SPEI-30 / 60 / 90 history", "weather-module forecasts"], "data")
    sv = box(ax, xs[2], r2, ww, bh, "Crop condition label (SVI)", ["SVI = (NDVI − normal) / spread",
                                                                    "per field and day of year"], "proc")
    arrow(ax, (a4[0] + a4[2] / 2, a4[1]), (a4[0] + a4[2] / 2, sv[1] + sv[3]))

    tw = (W - 2 * x0 - 2 * gap) / 3
    ts = [x0 + i * (tw + gap) for i in range(3)]
    la = box(ax, ts[0], r3, tw, bh, "Lag analysis", ["SVI vs lagged SPEI", "rainfed: 10-day lag"], "eval")
    fc = box(ax, ts[1], r3, tw, bh, "Crop forecast", ["SVI class, 15 / 30 days", "RF, XGBoost, onset warning"], "model")
    rp = box(ax, ts[2], r3, tw, bh, "Operational replay", ["alarm: SPEI-30 < −1", "confirm: SVI < −1", "→ stressed fields list"], "out")

    bus = r3 + bh + (r2 - r3 - bh) / 2
    cx = [b[0] + b[2] / 2 for b in (la, fc, rp)]
    for src in (wl, sv):
        ax.plot([src[0] + src[2] / 2] * 2, [src[1], bus], color=NAVY, lw=1.6)
    ax.plot([cx[0], cx[2]], [bus, bus], color=NAVY, lw=1.6)
    for x in cx:
        arrow(ax, (x, bus), (x, r3 + bh))

    ax.text(W / 2, 1.4, "Validated against AP / Karnataka drought orders 2023–2026 (comparison only)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=STYLE["eval"][1])

    p = OUT / "v2_satellite_slide.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    main()
