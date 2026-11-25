"""
Wide, large-font version of the system architecture for the presentation (one full slide).
Same content as v2_system in make_diagrams_v2.py, fewer words per box so it stays readable when projected.

    python final_report_materials/make_system_slide.py
Writes latex/figures/v2_system_slide.png (300 dpi, RGB).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent / "latex" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans"})
STYLE = {"model": ("#FFF3DE", "#C26A00"), "out": ("#FCE9E9", "#A83232"), "uav": ("#E3F4F4", "#1F7A7A"),
         "eval": ("#F1EAF8", "#6E4A9E")}
INK, NAVY, PURPLE = "#1E1E1E", "#1F3A5F", "#6E4A9E"
TS, LS, LH = 11.5, 10, 1.7      # title pt, line pt, line height (canvas units = 0.1 inch)
W, H = 80, 34


def box(ax, x, y, w, h, title, lines, kind):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fill, ec=edge, lw=1.6))
    block = 2.1 + len(lines) * LH
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=TS, fontweight="bold", color=edge)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 2.3 - i * LH, line, ha="center", va="top", fontsize=LS, color=INK)
    return (x, y, w, h)


def arrow(ax, p1, p2, color=NAVY):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color,
                                 shrinkA=0.5, shrinkB=0.5))


def main():
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    x0, gap = 5.0, 2.4
    bw = (W - 0.6 - x0 - 3 * gap) / 4
    xs = [x0 + i * (bw + gap) for i in range(4)]
    ky, ry, bh = 20.6, 7.0, 11.6

    ax.text(2.0, ky + bh / 2, "KHARIF\nJun – Nov", rotation=90, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=NAVY, linespacing=1.25)
    ax.text(2.0, ry + bh / 2, "RABI\nOct – Apr", rotation=90, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=PURPLE, linespacing=1.25)

    k1 = box(ax, xs[0], ky, bw, bh, "Weather forecast", ["AgERA5 weather", "SPEI-30 class", "7 / 15 / 30 days"], "model")
    k2 = box(ax, xs[1], ky, bw, bh, "Satellite check", ["Sentinel-2 fields", "is the crop", "stressed?"], "model")
    k3 = box(ax, xs[2], ky, bw, bh, "Stressed fields", ["confirmed drought", "list for drone"], "out")
    r1 = box(ax, xs[0], ry, bw, bh, "Season status", ["ERA5-Land soil", "deficit days (DD)", "+ MODIS greenness"], "model")
    r2 = box(ax, xs[1], ry, bw, bh, "Field early-drying", ["Random forest", "7 / 15 / 30 days", "soil-dry / wet tag"], "model")
    r3 = box(ax, xs[2], ry, bw, bh, "Fields to inspect", ["early warnings", "list for drone"], "out")
    u = box(ax, xs[3], ry, bw, ky + bh - ry, "UAV assessment",
            ["drone survey", "↓", "orthomosaic", "↓", "SegFormer-B2", "(7 classes)", "↓", "area in m²", "+ stress flag"],
            "uav")

    for a, b in ((k1, k2), (k2, k3), (r1, r2), (r2, r3)):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2))
    arrow(ax, (k3[0] + k3[2], k3[1] + k3[3] / 2), (u[0], k3[1] + k3[3] / 2))
    arrow(ax, (r3[0] + r3[2], r3[1] + r3[3] / 2), (u[0], r3[1] + r3[3] / 2))

    ax.add_patch(FancyBboxPatch((0.6, 0.6), W - 1.2, 4.2, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=STYLE["eval"][0], ec=STYLE["eval"][1], lw=1.2))
    ax.text(W / 2, 2.7, "Independent checks (not used in training):  drought orders  ·  rabi crop yields  ·  "
                        "UAV validation tiles", ha="center", va="center", fontsize=9.3, fontweight="bold",
            color=STYLE["eval"][1])

    p = OUT / "v2_system_slide.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    main()
