# Custom Human-in-the-Loop Annotation Tool

Source filename: annotate_balanced200.py

The project used a custom desktop annotation application
to manually refine automatically generated pseudo-masks.

## Technology

Python, Tkinter, Matplotlib TkAgg, OpenCV and NumPy.

## Workflow

1. Select an original UAV image.
2. Load the corresponding pseudo-mask by filename.
3. Convert pseudo-background label 0 to ignore label 255.
4. Divide the image into selectable non-overlapping tiles.
5. Compare original RGB imagery with the segmentation overlay.
6. Correct individual pixels using class-specific brushes.
7. Erase uncertain regions to ignore label 255.
8. Save the corrected image, mask and visualization.
9. Resume previously saved tiles when necessary.

## Configuration

Default tile size: 512.
Supported tile sizes: 128 to 2048.
Default brush radius: 20 pixels.
Brush radius range: 2 to 150 pixels.
Default overlay opacity: 50 percent.
Undo history: 50 strokes.

## Controls

Left drag: paint.
Right drag: erase to ignore.
Scroll or plus/minus: change brush size.
Square brackets: change opacity.
Z: undo.
R: reset.
S: save.
G or Escape: return to grid.
Keys 1 to 6: foreground semantic classes.
Key 0: ignore.

## Output

actual_images: RGB PNG tiles.
masks: single-channel class-index PNG masks.
overlays: JPG visualization.

Filename pattern:
source_image_r0000_c0000

## Important distinction

The annotation tool maps background label 0 to ignore 255.
The segmentation model nevertheless predicts seven classes.

This is semi-automatic annotation with human correction.
It is not equivalent to independent annotation from scratch.

The balanced_200 directory name does not establish
the actual number of annotated images.

No measured annotation-time savings are claimed.

The custom annotation process was used during dataset
preparation and is distinct from the pending independent
CVAT test annotation.