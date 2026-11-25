"""
Wide, large-font version of the rabi-module diagram for the presentation (one full slide).
Same content as v2_rabi in make_diagrams_v2.py, fewer words per box so it stays readable when projected.

    python final_report_materials/make_rabi_slide.py
Writes latex/figures/v2_rabi_slide.png (300 dpi, RGB).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from make_system_slide import NAVY, OUT, STYLE, arrow, box  # noqa: E402

STYLE.update({"data": ("#E6F0FA", "#2E6DB4")})
W, H = 80, 38


def main():
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    x0, gap, dw, mw = 0.6, 3.0, 22.0, 28.0
    ow = W - 2 * x0 - dw - mw - 2 * gap
    xd, xm, xo = x0, x0 + dw + gap, x0 + dw + mw + 2 * gap
    bh = 7.6
    ys = [26.8, 17.6, 8.4]

    for x, w, t in ((xd, dw, "Data"), (xm, mw, "Method"), (xo, ow, "Output")):
        ax.text(x + w / 2, 37.4, t, ha="center", va="top", fontsize=11, fontweight="bold", color=NAVY)

    rows = [
        (("ERA5-Land soil", ["root zone, 45 points", "1990–2023"]),
         ("Soil drought indicator", ["percentile vs 1990–2004", "DD = share of deficit days"]),
         ("Season status", ["per district", "warning by 15 Dec"])),
        (("MODIS NDVI, 250 m", ["cropland, 3 districts", "2000–2026"]),
         ("Greenness indicator", ["percentile vs 2000–2009", "VPCT = Dec–Feb mean"]),
         ("Confirmation", ["13 later seasons", "2010–2022"])),
        (("Sentinel-2 + AgERA5", ["1,350 random points", "2019–2026"]),
         ("Field early-drying model", ["Random forest, 39 features", "warns 7 / 15 / 30 days ahead"]),
         ("Field warnings", ["tagged by soil check", "→ list for drone"])),
    ]
    for y, (d, m, o) in zip(ys, rows):
        a = box(ax, xd, y, dw, bh, *d, "data")
        b = box(ax, xm, y, mw, bh, *m, "model")
        c = box(ax, xo, y, ow, bh, *o, "out")
        arrow(ax, (a[0] + a[2], y + bh / 2), (b[0], y + bh / 2))
        arrow(ax, (b[0] + b[2], y + bh / 2), (c[0], y + bh / 2))

    ax.add_patch(FancyBboxPatch((0.6, 0.6), W - 1.2, 5.6, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=STYLE["eval"][0], ec=STYLE["eval"][1], lw=1.2))
    ax.text(W / 2, 4.6, "Validation (not used in training): district rabi yields 1997–2022; loss season = yield ≤ −15 %", ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=STYLE["eval"][1])
    ax.text(W / 2, 2.2, "AUC for loss seasons:  soil DD 0.80 (by 15 Dec)  ·  MODIS 0.83  ·  rain-based SPEI 0.60",
            ha="center", va="center", fontsize=9.5, color="#1E1E1E")

    p = OUT / "v2_rabi_slide.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    main()
