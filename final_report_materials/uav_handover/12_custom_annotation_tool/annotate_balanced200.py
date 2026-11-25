# ANNOTATION TOOL - Fixed paths for D:\drought_validation_annotation\balanced_200
# Run with: python annotate_balanced200.py

import math
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
IMAGE_DIR = r"D:\drought_validation_annotation\balanced_200\original_images"
PSEUDO_MASK_DIR = r"D:\drought_validation_annotation\balanced_200\masks"
OVERLAY_DIR = r"D:\drought_validation_annotation\balanced_200\overlays"
OUTPUT_DIR = r"D:\drought_validation_annotation\balanced_200\final_clean"

# ------------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------------
CLASS_NAMES = ["background", "soil", "crop", "cotton_ball", "green_weed", "dry_weed", "drought"]
PALETTE_RGB = [
    (0, 0, 0), (139, 90, 43), (0, 180, 0), (255, 255, 255),
    (140, 255, 140), (210, 170, 90), (220, 30, 30),
]
IGNORE_IDX = 255
NUM_CLASSES = len(CLASS_NAMES)
CURSOR_COLOR_OUT = (255, 255, 0)
CURSOR_COLOR_IN = (0, 0, 0)
LEGEND_W_FRAC = 0.26

# Class 0 ("background") is treated as "no label / ignore" everywhere:
#  - key 0 and right-click paint IGNORE_IDX (255)
#  - if True, pixels labelled 0 in a loaded pseudo mask are converted to 255
#    so the final masks use ONE consistent meaning. Set False to keep them as 0.
MAP_PSEUDO_BACKGROUND_TO_IGNORE = True


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, col in enumerate(PALETTE_RGB):
        rgb[mask == i] = col
    rgb[mask == IGNORE_IDX] = (80, 80, 80)
    return rgb


def _build_overlay(img_rgb, mask, alpha):
    col = mask_to_rgb(mask)
    h, w = img_rgb.shape[:2]
    col_r = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img_rgb.astype(np.float32) * (1 - alpha) + col_r.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_cursor(base, cx, cy, radius):
    out = base.copy()
    cv2.circle(out, (cx, cy), max(radius, 1), CURSOR_COLOR_OUT, 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), max(radius, 1), CURSOR_COLOR_IN, 1, cv2.LINE_AA)
    return out


def load_image_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path, shape_hw):
    """Load a pseudo mask (or return a blank IGNORE mask)."""
    if path and Path(path).exists():
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[:2] != tuple(shape_hw):
                m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
            if MAP_PSEUDO_BACKGROUND_TO_IGNORE:
                m = m.copy()
                m[m == 0] = IGNORE_IDX
            return m
    return np.full(shape_hw, IGNORE_IDX, dtype=np.uint8)


def find_pseudo_mask(img_path, pseudo_dir):
    stem_lower = img_path.stem.lower()
    for folder in [pseudo_dir, pseudo_dir / "masks"]:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() == ".png" and f.stem.lower() == stem_lower:
                return f
    return None


def select_image_interactively(image_dir):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select image to annotate",
        initialdir=str(image_dir),
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def select_tile_size_interactively():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    tile_size = simpledialog.askinteger(
        "Tile Size", "Enter tile size:\n(512=default)",
        initialvalue=512, minvalue=128, maxvalue=2048,
    )
    root.destroy()
    return tile_size if tile_size else 512


# ------------------------------------------------------------------
# GRID VIEW
# ------------------------------------------------------------------
class GridView:
    def __init__(self, img_rgb, mask, tile_size, saved_set):
        self.img_rgb = img_rgb
        self.mask = mask
        self.ts = tile_size
        self.saved_set = saved_set
        self.H, self.W = img_rgb.shape[:2]
        self.rows = max(1, math.ceil(self.H / tile_size))
        self.cols = max(1, math.ceil(self.W / tile_size))
        self.selected = None
        self.quit = False

    def run(self):
        overlay = _build_overlay(self.img_rgb, self.mask, 0.35)
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.canvas.manager.set_window_title("GRID - click tile to edit | close to quit")
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.imshow(overlay)
        ax.set_title(
            f"Grid {self.rows}x{self.cols} | {len(self.saved_set)}/{self.rows * self.cols} saved "
            f"| click tile to edit | close to quit",
            fontsize=9, color="white", pad=4,
        )
        for c in range(self.cols + 1):
            ax.axvline(c * self.ts, color="cyan", lw=0.5, alpha=0.6)
        for r in range(self.rows + 1):
            ax.axhline(r * self.ts, color="cyan", lw=0.5, alpha=0.6)
        for (r, c) in self.saved_set:
            ax.add_patch(mpatches.Rectangle(
                (c * self.ts, r * self.ts), self.ts, self.ts,
                linewidth=2, edgecolor="lime", facecolor="lime", alpha=0.15))
        for r in range(self.rows):
            for c in range(self.cols):
                col = "lime" if (r, c) in self.saved_set else "cyan"
                ax.text(c * self.ts + 6, r * self.ts + 20, f"{r},{c}", color=col, fontsize=7, va="top")
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)

        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return
            c = int(event.xdata // self.ts)
            r = int(event.ydata // self.ts)
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.selected = (r, c)
                plt.close(fig)

        def on_close(event):
            if self.selected is None:
                self.quit = True

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("close_event", on_close)
        plt.tight_layout()
        plt.show()
        return self.selected


# ------------------------------------------------------------------
# TILE EDITOR
# ------------------------------------------------------------------
class TileEditor:
    def __init__(self, img_rgb, mask, row, col, label):
        self.img_rgb = img_rgb
        self.mask = mask.copy()
        self.orig_mask = mask.copy()
        self.row = row
        self.col = col
        self.label = label
        self.cur_class = 6
        self.brush = 20
        self.opacity = 0.50
        self.undo_stack = []      # one snapshot per stroke
        self.saved = False
        self._drawing = False
        self._erase = False
        self._last_cx = -1
        self._last_cy = -1
        self._base_ovl = None

    # ---- rendering -------------------------------------------------
    def _rebuild_base(self):
        self._base_ovl = _build_overlay(self.img_rgb, self.mask, self.opacity)

    def _frame_with_cursor(self, cx, cy):
        if self._base_ovl is None:
            self._rebuild_base()
        if cx < 0 or cy < 0:
            return self._base_ovl
        return _add_cursor(self._base_ovl, cx, cy, self.brush)

    def _show_cursor(self, cx, cy):
        self._ovl_im.set_data(self._frame_with_cursor(cx, cy))
        self._fig.canvas.draw_idle()

    def _full_refresh(self, cx=-1, cy=-1):
        self._rebuild_base()
        self._ovl_im.set_data(self._frame_with_cursor(cx, cy))
        self._update_legend()
        self._fig.canvas.draw_idle()

    def _update_legend(self):
        for cls, txt in self._cls_texts.items():
            is_cur = (cls == self.cur_class)
            txt.set_color("white" if is_cur else "#888888")
            txt.set_fontweight("bold" if is_cur else "normal")
        self._brush_txt.set_text(f"Brush: {self.brush}px")
        self._opacity_txt.set_text(f"Opacity: {int(self.opacity * 100)}%")
        self._status_txt.set_text("SAVED" if self.saved else "unsaved")
        self._status_txt.set_color("lime" if self.saved else "#60aaff")

    # ---- painting --------------------------------------------------
    def _paint_at(self, x, y):
        H, W = self.mask.shape
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < W and 0 <= iy < H):
            return
        Y, X = np.ogrid[:H, :W]
        circ = (X - ix) ** 2 + (Y - iy) ** 2 <= self.brush ** 2
        val = IGNORE_IDX if (self._erase or self.cur_class == 0) else self.cur_class
        self.mask[circ] = val

    # ---- events ----------------------------------------------------
    def _on_press(self, ev):
        if ev.inaxes != self._ax_ovl or ev.xdata is None:
            return
        self.undo_stack.append(self.mask.copy())   # one undo step per stroke
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self._drawing = True
        self._erase = (ev.button == 3)
        self._paint_at(ev.xdata, ev.ydata)
        self._rebuild_base()
        self._last_cx, self._last_cy = int(ev.xdata), int(ev.ydata)
        self._show_cursor(self._last_cx, self._last_cy)

    def _on_move(self, ev):
        if ev.inaxes != self._ax_ovl or ev.xdata is None:
            return
        cx, cy = int(ev.xdata), int(ev.ydata)
        if self._drawing:
            self._paint_at(ev.xdata, ev.ydata)
            self._rebuild_base()
        self._show_cursor(cx, cy)
        self._last_cx, self._last_cy = cx, cy

    def _on_release(self, ev):
        if not self._drawing:
            return
        self._drawing = False
        self.saved = False
        self._full_refresh(self._last_cx, self._last_cy)

    def _on_leave(self, ev):
        if self._base_ovl is not None:
            self._ovl_im.set_data(self._base_ovl)
            self._fig.canvas.draw_idle()

    def _on_scroll(self, ev):
        if ev.button == "up":
            self.brush = min(self.brush + 3, 150)
        else:
            self.brush = max(self.brush - 3, 2)
        self._show_cursor(self._last_cx, self._last_cy)
        self._update_legend()
        self._fig.canvas.draw_idle()

    def _on_key(self, ev):
        k = ev.key
        if k in ("g", "escape", "q"):
            plt.close(self._fig)
        elif k == "s":
            self.saved = True
            self._full_refresh()
            plt.pause(0.1)
            plt.close(self._fig)
        elif k == "z" and self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.saved = False
            self._full_refresh(self._last_cx, self._last_cy)
        elif k == "r":
            self.mask = self.orig_mask.copy()
            self.undo_stack.clear()
            self.saved = False
            self._full_refresh(self._last_cx, self._last_cy)
        elif k in ("-", "_"):
            self.brush = max(self.brush - 3, 2)
            self._show_cursor(self._last_cx, self._last_cy)
            self._update_legend()
            self._fig.canvas.draw_idle()
        elif k in ("+", "="):
            self.brush = min(self.brush + 3, 150)
            self._show_cursor(self._last_cx, self._last_cy)
            self._update_legend()
            self._fig.canvas.draw_idle()
        elif k == "[":
            self.opacity = max(self.opacity - 0.05, 0.0)
            self._full_refresh(self._last_cx, self._last_cy)
        elif k == "]":
            self.opacity = min(self.opacity + 0.05, 1.0)
            self._full_refresh(self._last_cx, self._last_cy)
        elif k in [str(i) for i in range(0, NUM_CLASSES)]:
            self.cur_class = int(k)
            self._update_legend()
            self._fig.canvas.draw_idle()

    # ---- window ----------------------------------------------------
    def run(self):
        self._rebuild_base()
        fig = plt.figure(figsize=(16, 7))
        fig.patch.set_facecolor("#1a1a1a")
        self._fig = fig
        fig.canvas.manager.set_window_title(
            f"TILE ({self.row},{self.col}) | s=save g=back z=undo r=reset 1-6=class 0=ignore")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, LEGEND_W_FRAC],
                              left=0.005, right=0.995, top=0.95, bottom=0.02, wspace=0.03)
        ax_img = fig.add_subplot(gs[0])
        ax_ovl = fig.add_subplot(gs[1])
        ax_leg = fig.add_subplot(gs[2])
        self._ax_ovl = ax_ovl
        for ax in (ax_img, ax_ovl, ax_leg):
            ax.set_facecolor("#1a1a1a")
            ax.axis("off")

        ax_img.imshow(self.img_rgb)
        ax_img.set_title("Original", color="#aaa", fontsize=8, pad=2)
        self._ovl_im = ax_ovl.imshow(self._base_ovl)
        ax_ovl.set_title("Edit: left=paint right=erase scroll=brush", color="#aaa", fontsize=8, pad=2)

        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(0, 1)
        ax_leg.text(0.05, 0.97, f"Tile ({self.row},{self.col})", color="#ffd740",
                    fontsize=9, fontweight="bold", va="top", transform=ax_leg.transAxes)
        self._status_txt = ax_leg.text(0.05, 0.925, "unsaved", color="#60aaff",
                                       fontsize=8, va="top", transform=ax_leg.transAxes)
        ax_leg.text(0.05, 0.875, "CLASSES", color="#888", fontsize=7, va="top",
                    transform=ax_leg.transAxes)

        # Legend: classes 1..6, then [0] ignore (class 0 = no label)
        self._cls_texts = {}
        y_step, y0 = 0.054, 0.835
        row_i = 0
        for i in range(1, NUM_CLASSES):
            norm = tuple(v / 255 for v in PALETTE_RGB[i])
            ax_leg.add_patch(mpatches.FancyBboxPatch(
                (0.03, y0 - row_i * y_step - 0.011), 0.06, 0.033,
                boxstyle="round,pad=0.005", facecolor=norm, edgecolor="#444",
                transform=ax_leg.transAxes, clip_on=False))
            self._cls_texts[i] = ax_leg.text(0.13, y0 - row_i * y_step, f"[{i}] {CLASS_NAMES[i]}",
                                             color="#888", fontsize=7, va="center",
                                             transform=ax_leg.transAxes)
            row_i += 1
        ax_leg.add_patch(mpatches.FancyBboxPatch(
            (0.03, y0 - row_i * y_step - 0.011), 0.06, 0.033,
            boxstyle="round,pad=0.005", facecolor=(80 / 255,) * 3, edgecolor="#444",
            transform=ax_leg.transAxes, clip_on=False))
        self._cls_texts[0] = ax_leg.text(0.13, y0 - row_i * y_step, "[0] ignore",
                                         color="#888", fontsize=7, va="center",
                                         transform=ax_leg.transAxes)
        yi = y0 - (row_i + 1) * y_step

        controls = ["CONTROLS", "left drag=paint", "right drag=erase", "scroll/-/+=brush",
                    "[/]=opacity", "z=undo stroke", "r=reset", "s=save+back", "g/Esc=back",
                    "1-6=class", "0=ignore"]
        yc = yi - 0.02
        for line in controls:
            ax_leg.text(0.05, yc, line, color="#555", fontsize=6.2, va="top",
                        transform=ax_leg.transAxes)
            yc -= 0.045
        self._brush_txt = ax_leg.text(0.05, yc - 0.01, f"Brush: {self.brush}px", color="#c0b030",
                                      fontsize=7.5, fontweight="bold", va="top",
                                      transform=ax_leg.transAxes)
        self._opacity_txt = ax_leg.text(0.05, yc - 0.065, f"Opacity: {int(self.opacity * 100)}%",
                                        color="#30b0c0", fontsize=7.5, fontweight="bold",
                                        va="top", transform=ax_leg.transAxes)
        self._update_legend()

        canvas = fig.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("axes_leave_event", self._on_leave)
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("key_press_event", self._on_key)
        plt.show()
        return self.saved, self.mask


# ------------------------------------------------------------------
# TILE ANNOTATOR
# ------------------------------------------------------------------
class TileAnnotator:
    def __init__(self, img_path, mask_path, tile_size, out_dir):
        self.img_path = img_path
        self.ts = tile_size
        self.out_dir = out_dir
        self.mask_dir = out_dir / "masks"
        self.ovl_dir = out_dir / "overlays"
        self.img_dir = out_dir / "actual_images"
        for d in (self.mask_dir, self.ovl_dir, self.img_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.img_rgb = load_image_rgb(img_path)
        if self.img_rgb is None:
            raise RuntimeError(f"Cannot read: {img_path}")
        self.H, self.W = self.img_rgb.shape[:2]
        self.rows = max(1, math.ceil(self.H / tile_size))
        self.cols = max(1, math.ceil(self.W / tile_size))
        self.full_mask = load_mask(mask_path, (self.H, self.W))

        print(f"\nImage: {img_path.name} ({self.W}x{self.H})")
        print(f"Grid : {self.rows}x{self.cols} tiles at {tile_size}px")
        print(f"Output: {out_dir}")

        self.saved_set = set()
        self._load_saves()

    def _load_saves(self):
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.mask_dir / f"{self.img_path.stem}_r{r:04d}_c{c:04d}.png"
                if p.exists():
                    tile = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if tile is not None:
                        y0, x0 = r * self.ts, c * self.ts
                        th, tw = tile.shape[:2]
                        self.full_mask[y0:y0 + th, x0:x0 + tw] = tile
                    self.saved_set.add((r, c))
        if self.saved_set:
            print(f"Resumed: {len(self.saved_set)} tiles already saved.")

    def _crop(self, r, c):
        y0 = r * self.ts
        y1 = min(y0 + self.ts, self.H)
        x0 = c * self.ts
        x1 = min(x0 + self.ts, self.W)
        return (self.img_rgb[y0:y1, x0:x1].copy(),
                self.full_mask[y0:y1, x0:x1].copy(), y0, x0)

    def _save_tile(self, r, c, mask_tile, y0, x0):
        stem = f"{self.img_path.stem}_r{r:04d}_c{c:04d}"
        th, tw = mask_tile.shape[:2]
        self.full_mask[y0:y0 + th, x0:x0 + tw] = mask_tile

        mask_path = self.mask_dir / f"{stem}.png"
        image_path = self.img_dir / f"{stem}.png"
        overlay_path = self.ovl_dir / f"{stem}.jpg"

        img_tile = self.img_rgb[y0:y0 + th, x0:x0 + tw]
        overlay = _build_overlay(img_tile, mask_tile, 0.45)

        ok_mask = cv2.imwrite(str(mask_path), mask_tile)
        ok_img = cv2.imwrite(str(image_path), cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
        ok_ovl = cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not all([ok_mask, ok_img, ok_ovl]):
            raise RuntimeError(f"Failed to save tile: {stem}")

        self.saved_set.add((r, c))
        print(f"\nSuccessfully saved tile: {stem}")
        print(f"Image:   {image_path}")
        print(f"Mask:    {mask_path}")
        print(f"Overlay: {overlay_path}")

    def run(self):
        while True:
            gv = GridView(self.img_rgb, self.full_mask, self.ts, self.saved_set)
            rc = gv.run()
            if gv.quit or rc is None:
                print(f"\nDone. {len(self.saved_set)} tiles saved.")
                break
            r, c = rc
            img_crop, mask_crop, y0, x0 = self._crop(r, c)
            editor = TileEditor(img_crop, mask_crop, r, c, f"{self.img_path.stem} [{r},{c}]")
            saved, final_mask = editor.run()
            if saved:
                self._save_tile(r, c, final_mask, y0, x0)


# ------------------------------------------------------------------
# LAUNCH
# ------------------------------------------------------------------
def main():
    print("Paths configured:")
    print(f"  Images      : {IMAGE_DIR}")
    print(f"  Pseudo masks: {PSEUDO_MASK_DIR}")
    print(f"  Output      : {OUTPUT_DIR}")

    print("\n" + "=" * 50)
    print("ANNOTATION TOOL - balanced_200")
    print("=" * 50)

    print("\nStep 1: Select an image file...")
    img_path = select_image_interactively(Path(IMAGE_DIR))
    if img_path is None:
        print("No image selected. Exiting.")
        return
    print(f"Selected: {img_path.name}")

    print("\nStep 2: Choose tile size...")
    tile_size = select_tile_size_interactively()
    print(f"Tile size: {tile_size}px")

    out_dir = Path(OUTPUT_DIR)
    mask_path = find_pseudo_mask(img_path, Path(PSEUDO_MASK_DIR))
    if mask_path:
        print(f"Pseudo mask found: {mask_path.name}")
    else:
        print("No pseudo mask found - starting with blank IGNORE mask")

    print("\nLaunching annotation window...")
    try:
        ann = TileAnnotator(img_path=img_path, mask_path=mask_path,
                            tile_size=tile_size, out_dir=out_dir)
        ann.run()

        # Preview only tiles from the image just annotated
        ovl_files = sorted((out_dir / "overlays").glob(f"{img_path.stem}_r*_c*.jpg"))
        if ovl_files:
            samples = random.sample(ovl_files, min(4, len(ovl_files)))
            fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 5))
            if len(samples) == 1:
                axes = [axes]
            for ax, p in zip(axes, samples):
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(p.stem, fontsize=8)
                ax.axis("off")
            plt.suptitle(f"Corrected Tiles Preview ({len(ovl_files)} total for {img_path.stem})",
                         fontsize=12)
            plt.tight_layout()
            plt.show()
            print(f"\nDone. {len(ovl_files)} tiles saved for this image.")
    except Exception as e:
        print(f"Error: {e}")
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(e))
        root.destroy()


if __name__ == "__main__":
    main()