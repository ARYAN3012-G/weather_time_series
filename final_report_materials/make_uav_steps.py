"""
Step images for the UAV module (report + slides), made from real project data:
  - a real drone photo and its automatic pseudo-label (Desktop/Annotation/balanced_200_set3)
  - real screenshots of the custom annotation tool (Desktop/Annotation/Output)
  - the 9 input channels, computed with the formulas of src/preprocessing/indices.py + tensor_builder.py
  - tiling (1024 px, stride 768) and Hann weights as in src/inference/tile_v11.py (clamp 0.1 per axis)

    python final_report_materials/make_uav_steps.py
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ANN = Path(r"C:\Users\aryan\OneDrive\Desktop\Annotation")
OUT = Path(__file__).resolve().parent / "latex" / "figures"
PHOTO = ANN / "balanced_200_set3" / "original_images" / "DJI_20251110160727_0926_D.jpg"
MASK = ANN / "balanced_200_set3" / "masks" / "DJI_20251110160727_0926_D.png"
TOOL = ANN / "Output" / "TILE_(0,0)___s=save_g=back_z=undo_r=reset_1-7=class_0=ignore.png"
plt.rcParams.update({"font.family": "DejaVu Sans"})
PALETTE = np.array([[0, 0, 0], [139, 90, 43], [0, 180, 0], [255, 255, 255], [140, 255, 140], [210, 170, 90],
                    [220, 30, 30]], np.uint8)
NAMES = ["background", "soil", "crop", "cotton boll", "green weed", "dry weed", "drought"]


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    Image.open(p).convert("RGB").save(p)
    plt.close(fig)
    print("wrote", name)


def colour(mask):
    out = np.full((*mask.shape, 3), 110, np.uint8)            # ignore (255) = grey
    for c in range(7):
        out[mask == c] = PALETTE[c]
    return out


def labels_figure(rgb, mask):
    tool = np.asarray(Image.open(TOOL).convert("RGB"))
    tool = tool[25:615, 15:1225]                                # original + edit panels of the tool window
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1.78, 1.78, 2.05]))
    small = cv2.resize(rgb, (1008, 567), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (1008, 567), interpolation=cv2.INTER_NEAREST)
    ax[0].imshow(small)
    ax[0].set_title("1. Drone photo (8064 × 4536 px)", fontsize=12, fontweight="bold")
    blend = (0.45 * small + 0.55 * colour(msmall)).astype(np.uint8)
    ax[1].imshow(blend)
    ax[1].set_title("2. Automatic pseudo-label (grey = ignore)", fontsize=12, fontweight="bold")
    ax[2].imshow(tool)
    ax[2].set_title("3. Correction in the annotation tool", fontsize=12, fontweight="bold")
    for a in ax:
        a.axis("off")
    handles = [Rectangle((0, 0), 1, 1, fc=PALETTE[c] / 255, ec="#555") for c in range(1, 7)]
    handles.append(Rectangle((0, 0), 1, 1, fc=(110 / 255,) * 3, ec="#555"))
    fig.legend(handles, NAMES[1:] + ["ignore"], loc="lower center", ncol=7, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.0)
    save(fig, "uav_step_labels.png")


def channels_figure(rgb):
    crop = rgb[1500:2012, 3300:3812]                            # one 512 x 512 tile of the photo
    f = crop.astype(np.float32) / 255
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    eps = 1e-6
    ch = [("RGB (3 channels)", crop, None),
          ("ExG / 2", (2 * g - r - b) / 2, "RdYlGn"),
          ("ExGR / 3", ((2 * g - r - b) - (1.4 * r - g)) / 3, "RdYlGn"),
          ("−CIVE / 0.811", -(0.441 * r - 0.811 * g + 0.385 * b) / 0.811, "RdYlGn"),
          ("GLI", np.clip((2 * g - r - b) / (2 * g + r + b + eps), -1, 1), "RdYlGn"),
          ("NGRDI", np.clip((g - r) / (g + r + eps), -1, 1), "RdYlGn"),
          ("HSV value V", cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[..., 2] / 255 * 2 - 1, "gray")]
    fig, ax = plt.subplots(1, 7, figsize=(17, 3.0))
    for a, (t, im, cm) in zip(ax, ch):
        if cm is None:
            a.imshow(im)
        else:
            lo, hi = np.percentile(im, [2, 98])
            a.imshow(im, cmap=cm, vmin=lo, vmax=hi)
        a.set_title(t, fontsize=12.5, fontweight="bold")
        a.axis("off")
    fig.tight_layout(w_pad=0.6)
    save(fig, "uav_step_channels.png")


def stitching_figure(rgb):
    H, W = rgb.shape[:2]
    T, S = 1024, 768
    small = cv2.resize(rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    fig = plt.figure(figsize=(16, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.55], wspace=0.18)
    # (a) tile grid
    a = fig.add_subplot(gs[0])
    a.imshow(small)
    for y in range(0, H - T + S, S):
        for x in range(0, W - T + S, S):
            a.add_patch(Rectangle((x / 4, y / 4), T / 4, T / 4, fill=False, ec="white", lw=0.7, alpha=0.75))
    for (x, y, c) in [(768, 768, "#FFD400"), (1536, 768, "#00E5FF")]:
        a.add_patch(Rectangle((x / 4, y / 4), T / 4, T / 4, fill=False, ec=c, lw=2.6))
    a.add_patch(Rectangle((1536 / 4, 768 / 4), 256 / 4, T / 4, fc="#FF3B30", alpha=0.35, ec="none"))
    a.set_title("(a) 1024 px tiles, stride 768 → 256 px overlap (red)", fontsize=12, fontweight="bold")
    a.axis("off")
    # (b) Hann weights along one row, exactly as in tile_v11.py
    b = fig.add_subplot(gs[1])
    w1 = np.clip(np.hanning(T), 0.1, None)
    xs = np.arange(0, 3 * S + T)
    tot = np.zeros(len(xs))
    for k, c in zip(range(4), ["#1f77b4", "#FFB000", "#00A3C4", "#7A5195"]):
        seg = np.zeros(len(xs))
        seg[k * S:k * S + T] = w1
        tot += seg
        b.plot(xs, seg, color=c, lw=1.8, label="tile weight" if k == 0 else None)
    b.plot(xs, tot, color="#C0392B", lw=1.6, ls="--", label="sum of weights")
    b.set_xlabel("pixel position along a row", fontsize=10.5)
    b.set_ylabel("Hann weight", fontsize=10.5)
    b.set_title("(b) Overlaps are blended by Hann weights", fontsize=12, fontweight="bold")
    b.legend(fontsize=9.5, loc="upper right", frameon=False)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.set_ylim(0, 1.45)
    # (c) rolling band
    c = fig.add_subplot(gs[2])
    c.imshow(small)
    y0 = 1536
    c.add_patch(Rectangle((0, 0), W / 4, y0 / 4, fc="white", alpha=0.55, ec="none"))
    c.add_patch(Rectangle((0, y0 / 4), W / 4, T / 4, fill=False, ec="#FFD400", lw=3))
    c.text(W / 8, y0 / 8, "rows finished: class label written,\nprobabilities freed", ha="center", va="center",
           fontsize=10.5, fontweight="bold", color="#1E1E1E")
    c.text(W / 8, (y0 + T / 2) / 4, "active band (1024 rows):\nonly these probabilities in memory", ha="center",
           va="center", fontsize=10.5, fontweight="bold", color="white",
           bbox=dict(fc="#1F3A5F", ec="none", alpha=0.8, pad=3))
    c.set_title("(c) Rolling accumulator: band moves down", fontsize=12, fontweight="bold")
    c.axis("off")
    save(fig, "uav_step_stitching.png")


def main():
    rgb = cv2.cvtColor(cv2.imread(str(PHOTO)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(MASK), cv2.IMREAD_UNCHANGED)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    labels_figure(rgb, mask)
    channels_figure(rgb)
    stitching_figure(rgb)


if __name__ == "__main__":
    main()
