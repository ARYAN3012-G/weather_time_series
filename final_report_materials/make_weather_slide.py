"""
Wide, large-font version of the weather-module diagram for the presentation (one full slide).
Same content as v2_weather in make_diagrams_v2.py, fewer words per box so it stays readable when projected.

    python final_report_materials/make_weather_slide.py
Writes latex/figures/v2_weather_slide.png (300 dpi, RGB).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from make_system_slide import OUT, STYLE, arrow, box  # noqa: E402

STYLE.update({"data": ("#E6F0FA", "#2E6DB4"), "proc": ("#E9F5E9", "#2F7D32")})
W, H = 80, 34


def main():
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    x0, gap = 0.6, 2.4
    bw = (W - 2 * x0 - 3 * gap) / 4
    xs = [x0 + i * (bw + gap) for i in range(4)]
    ty, by, bh = 21.6, 7.4, 11.4

    t1 = box(ax, xs[0], ty, bw, bh, "AgERA5 weather", ["daily, 1979–2026", "6 independent cells", "rain, ET₀, T, RH"], "data")
    t2 = box(ax, xs[1], ty, bw, bh, "Water balance", ["D = rain − ET₀", "summed over 30 days"], "proc")
    t3 = box(ax, xs[2], ty, bw, bh, "SPEI-30", ["log-logistic fit", "on 1979–2011", "then frozen"], "proc")
    t4 = box(ax, xs[3], ty, bw, bh, "Drought class", ["Healthy ≥ 0", "Moderate −1.5 … 0", "Severe < −1.5"], "proc")
    b1 = box(ax, xs[0], by, bw, bh, "Input features", ["30-day window", "9 weather variables", "SPEI history"], "proc")
    b2 = box(ax, xs[1], by, bw, bh, "Physics ensemble", ["water balance with", "each 1979–2011 year", "→ class probabilities"], "model")
    b3 = box(ax, xs[2], by, bw, bh, "Models (5 seeds)", ["RF, XGBoost", "LSTM, Transformer", "soft voting"], "model")
    b4 = box(ax, xs[3], by, bw, bh, "Evaluation", ["purged time split", "test 2019–2026", "vs 4 baselines"], "eval")

    for a, b in ((t1, t2), (t2, t3), (t3, t4), (b1, b2), (b2, b3), (b3, b4)):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2))
    for a, b in ((t1, b1), (t2, b2), (t4, b4)):
        arrow(ax, (a[0] + a[2] / 2, a[1]), (b[0] + b[2] / 2, b[1] + b[3]))
    ax.text(xs[3] + bw / 2 - 0.8, (by + bh + ty) / 2, "target: class at t+h", ha="right", va="center",
            fontsize=9, style="italic", color="#1F3A5F")

    ax.add_patch(FancyBboxPatch((0.6, 0.6), W - 1.2, 4.2, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=STYLE["out"][0], ec=STYLE["out"][1], lw=1.2))
    ax.text(W / 2, 2.7, "Output: drought class per site, 7 / 15 / 30 days ahead  →  alarm for the satellite module",
            ha="center", va="center", fontsize=10, fontweight="bold", color=STYLE["out"][1])

    p = OUT / "v2_weather_slide.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    main()
