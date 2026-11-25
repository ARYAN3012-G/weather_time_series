"""
Compact, large-font version of the nine-channel SegFormer-B2 diagram for the presentation.
Same content as v2_uav_model in make_diagrams_v2.py, grouped into fewer boxes so it stays readable.

    python final_report_materials/make_uav_model_slide.py
Writes latex/figures/v2_uav_model_slide.png (300 dpi, RGB).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import make_system_slide as mss  # noqa: E402
from make_system_slide import OUT, STYLE, arrow, box  # noqa: E402

STYLE.update({"data": ("#E6F0FA", "#2E6DB4"), "proc": ("#E9F5E9", "#2F7D32")})
mss.TS, mss.LS = 11, 9.2
W, H = 57, 34


def main():
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    gap = 2.6
    bw = (W - 1.0 - 2 * gap) / 3
    xs = [0.5 + i * (bw + gap) for i in range(3)]
    r1, r2, bh = 21.6, 8.6, 10.6

    i_ = box(ax, xs[0], r1, bw, bh, "Input tile", ["9 channels", "512 × 512 px", "RGB + 6 indices"], "data")
    pe = box(ax, xs[1], r1, bw, bh, "Patch embed", ["7×7 conv, stride 4", "9 → 64 channels"], "proc")
    en = box(ax, xs[2], r1, bw, bh, "MiT-B2 encoder", ["4 stages, H/4 → H/32", "64, 128, 320, 512 ch", "self-attention"], "model")
    de = box(ax, xs[2], r2, bw, bh, "MLP decoder", ["each stage → 768 ch", "upsample to H/4", "concatenate + fuse"], "proc")
    ou = box(ax, xs[1], r2, bw, bh, "Output", ["7-class logits", "bilinear ×4", "logit adjust, argmax"], "out")
    mk = box(ax, xs[0], r2, bw, bh, "Class map", ["soil, crop, cotton,", "green / dry weed,", "drought, background"], "eval")

    arrow(ax, (i_[0] + bw, r1 + bh / 2), (pe[0], r1 + bh / 2))
    arrow(ax, (pe[0] + bw, r1 + bh / 2), (en[0], r1 + bh / 2))
    arrow(ax, (en[0] + bw / 2, r1), (de[0] + bw / 2, r2 + bh))
    arrow(ax, (de[0], r2 + bh / 2), (ou[0] + bw, r2 + bh / 2))
    arrow(ax, (ou[0], r2 + bh / 2), (mk[0] + bw, r2 + bh / 2))

    ax.add_patch(FancyBboxPatch((0.5, 0.5), W - 1.0, 5.6, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc="#FFF3DE", ec="#C26A00", lw=1.2))
    ax.text(W / 2, 4.5, "Weight widening, 3 → 9 input channels (ImageNet-pretrained)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#C26A00")
    ax.text(W / 2, 2.1, "W[:, 0:3] = W_old / 3      W[:, 3:9] = 0  (new channels start at zero)", ha="center",
            va="center", fontsize=9.2, color="#1E1E1E")

    p = OUT / "v2_uav_model_slide.png"
    fig.savefig(p, dpi=300, facecolor="white")
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    main()
