"""
Wide, large-font version of the 15-step UAV pipeline for the presentation (one full slide).
Same steps as v2_uav_pipeline in make_diagrams_v2.py: Phase A (left) and Phase B (right), shorter lines.

    python final_report_materials/make_uav_pipeline_slide.py
Writes latex/figures/v2_uav_pipeline_slide.png (300 dpi, RGB).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import make_system_slide as mss  # noqa: E402
from make_system_slide import OUT, STYLE, arrow, box  # noqa: E402

mss.TS, mss.LS = 10.2, 9.2

STYLE.update({"data": ("#E6F0FA", "#2E6DB4"), "proc": ("#E9F5E9", "#2F7D32")})
W, H = 80, 38
ORANGE, TEAL = "#C26A00", "#1F7A7A"


def band(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=1.0", fc=color + "12",
                                ec=color, lw=1.0, ls=(0, (4, 3))))
    ax.text(x + 0.8, y + h - 0.5, label, ha="left", va="top", fontsize=10, fontweight="bold", color=color)


def main():
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    band(ax, 0.3, 0.4, 39.0, 37.2, "A.  Build labels and train the model", ORANGE)
    band(ax, 40.7, 0.4, 39.0, 37.2, "B.  Process one field (with the trained model)", TEAL)

    bw, bh = 17.0, 6.4
    ys = [26.6, 18.4, 10.2, 2.0]
    ax1, ax2 = 1.3, 21.3
    bx1, bx2 = 41.7, 61.7

    A = [("1. Drone photos", ["RGB images", "14 surveyed fields"], "data"),
         ("2. Patch classifier", ["DINOv2 ViT-B/14", "macro-F1 0.50–0.60"], "model"),
         ("3. Pseudo-labels", ["sliding window + CRF", "conf. < 0.55 → ignore"], "proc"),
         ("4. Annotation tool", ["brush correction", "→ 279 clean tiles"], "proc"),
         ("5. Training mix", ["clean + pseudo +", "synthetic composites"], "proc"),
         ("6. 9-channel input", ["RGB + 6 vegetation", "indices"], "proc"),
         ("7. SegFormer-B2", ["focal + Dice + Lovász", "best val. mIoU"], "model"),
         ("8. Logit adjustment", ["drought −1.2", "dry weed −0.4"], "model")]
    a = [box(ax, ax1, ys[i], bw, bh, *A[i]) for i in range(4)]
    a += [box(ax, ax2, ys[3 - i], bw, bh, *A[4 + i]) for i in range(4)]
    for i in range(3):
        arrow(ax, (ax1 + bw / 2, ys[i]), (ax1 + bw / 2, ys[i + 1] + bh))
    arrow(ax, (ax1 + bw, ys[3] + bh / 2), (ax2, ys[3] + bh / 2))
    for i in range(3):
        arrow(ax, (ax2 + bw / 2, ys[3 - i] + bh), (ax2 + bw / 2, ys[2 - i]))

    b9 = box(ax, bx1, ys[0], bx2 + bw - bx1, bh, "9. Orthomosaic", ["Agisoft Metashape, up to 2.86 billion pixels"], "data")
    B1 = [("10. Tiled prediction", ["1024 px tiles", "stride 768"], "model"),
          ("11. Rolling stitching", ["Hann-weighted softmax", "1.5 GB, not 80 GB"], "proc"),
          ("12. Post-processing", ["border, glare, shadow", "small regions, harvest"], "proc")]
    B2 = [("13. Area estimate", ["pixels × GSD²", "→ m² per class"], "proc"),
          ("14. Stress decision", ["D > 0.15 or", "(N > 0.10 and C > 0.25)"], "eval"),
          ("15. Field report", ["mask, overlay,", "JSON, PDF"], "out")]
    b1 = [box(ax, bx1, ys[1 + i], bw, bh, *B1[i]) for i in range(3)]
    b2 = [box(ax, bx2, ys[3 - i], bw, bh, *B2[i]) for i in range(3)]
    arrow(ax, (bx1 + bw / 2, ys[0]), (bx1 + bw / 2, ys[1] + bh))
    for i in range(2):
        arrow(ax, (bx1 + bw / 2, ys[1 + i]), (bx1 + bw / 2, ys[2 + i] + bh))
    arrow(ax, (bx1 + bw, ys[3] + bh / 2), (bx2, ys[3] + bh / 2))
    for i in range(2):
        arrow(ax, (bx2 + bw / 2, ys[3 - i] + bh), (bx2 + bw / 2, ys[2 - i]))

    # trained model: step 8 (A, top right) -> step 10 (B, second row left)
    y8, y10 = ys[0] + bh / 2, ys[1] + bh / 2
    xm = 40.0
    ax.plot([ax2 + bw, xm], [y8, y8], color=ORANGE, lw=1.6)
    ax.plot([xm, xm], [y8, y10], color=ORANGE, lw=1.6)
    arrow(ax, (xm, y10), (bx1, y10), color=ORANGE)
    del a, b9, b1, b2

    p = OUT / "v2_uav_pipeline_slide.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    main()
