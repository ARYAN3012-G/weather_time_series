
"""Block-wise postprocessing with global connectivity."""

import cv2
import numpy as np
import tempfile

from src.preprocessing import masks as original_masks
import importlib
original_post = importlib.import_module("src.inference.postprocess")
from src.preprocessing.disk_components import (
    strip_components,
    component_mask
)


def blocks(height, rows, halo=0):

    for y in range(0, height, rows):

        end = min(y + rows, height)
        lo = max(0, y - halo)
        hi = min(height, end + halo)

        yield y, end, lo, hi


def select_components(binary, selector, rows=256):

    with tempfile.TemporaryDirectory(
        prefix="drought_boundary_",
        dir="/content"
    ) as work:

        result = boundary_cc.analyze(
            binary,
            strip_rows=rows,
            work_dir=work
        )

        return boundary_cc.selected_mask(
            result, selector
        )


def valid_field_mask(
    rgb,
    dark_threshold=10,
    saturation_threshold=15,
    min_border_component_area=500,
    rows=256
):

    H, W = rgb.shape[:2]

    candidate = np.empty(
        (H, W), dtype=np.uint8
    )

    for y, end, _, _ in blocks(H, rows):

        block = rgb[y:end]

        maximum = block.max(axis=2)
        minimum = block.min(axis=2)

        saturation = (
            maximum.astype(np.int16)
            - minimum.astype(np.int16)
        )

        dark = (
            (maximum <= dark_threshold)
            & (saturation <= saturation_threshold)
        )

        white = (
            (maximum >= 248)
            & (saturation <= 8)
        )

        candidate[y:end] = dark | white

    bg = select_components(
        candidate,
        lambda areas, borders: (
            borders
            & (areas >= min_border_component_area)
        ),
        rows
    )

    del candidate

    result = np.empty(
        (H, W), dtype=bool
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    for y, end, lo, hi in blocks(H, rows, 4):

        dilated = cv2.dilate(
            bg[lo:hi].astype(np.uint8),
            kernel,
            iterations=2
        )

        result[y:end] = ~dilated[
            y-lo:end-lo
        ].astype(bool)

    return result


def compute_aux_masks(
    rgb,
    *,
    glare_v=0.95,
    glare_s=0.10,
    shadow_v=0.15,
    shadow_s=0.20,
    field_dark=10,
    field_sat=15,
    field_min_area=500,
    rows=256
):

    H, W = rgb.shape[:2]

    valid = valid_field_mask(
        rgb,
        field_dark,
        field_sat,
        field_min_area,
        rows
    )

    glare = np.empty((H, W), dtype=bool)
    shadow = np.empty((H, W), dtype=bool)

    for y, end, _, _ in blocks(H, rows):

        glare[y:end] = original_masks.glare_mask(
            rgb[y:end], glare_v, glare_s
        )

        shadow[y:end] = original_masks.shadow_mask(
            rgb[y:end], shadow_v, shadow_s
        )

    return {
        "valid_field": valid,
        "glare": glare,
        "shadow": shadow
    }


def soften_problematic_pixels(
    class_map,
    rgb,
    aux_masks=None,
    rows=256
):

    masks = (
        compute_aux_masks(rgb, rows=rows)
        if aux_masks is None
        else aux_masks
    )

    H, W = class_map.shape

    result = class_map.copy()

    for y, end, lo, hi in blocks(H, rows, 4):

        problem = (
            masks["glare"][y:end]
            | masks["shadow"][y:end]
        )

        if not problem.any():
            continue

        best = np.zeros(
            (end-y, W), dtype=np.uint8
        )

        target = result[y:end]

        for cls in range(original_post.NUM_CLASSES):

            indicator = (
                class_map[lo:hi] == cls
            ).astype(np.uint8)

            count = cv2.boxFilter(
                indicator,
                ddepth=-1,
                ksize=(9, 9),
                normalize=False
            )[y-lo:end-lo]

            dominates = (
                (count > best) & problem
            )

            target[dominates] = cls

            np.maximum(
                best, count, out=best
            )

    return result


def morphological_cleanup(class_map, rows=256):

    H, W = class_map.shape

    result = class_map.copy()

    noise = np.zeros(
        (H, W), dtype=bool
    )

    for name, minimum in (
        original_post.MIN_COMPONENT_SIZES.items()
    ):

        if minimum <= 0:
            continue

        cls = original_post.CLASS_TO_IDX.get(name)

        if cls is None:
            continue

        binary = (class_map == cls)

        if not binary.any():
            continue

        small = select_components(
            binary,
            lambda areas, borders: (
                (areas > 0)
                & (areas < minimum)
            ),
            rows
        )

        noise |= small

        del binary, small

    if not noise.any():
        return result

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (15, 15)
    )

    for y, end, lo, hi in blocks(H, rows, 7):

        target = result[y:end]

        for cls in range(
            1, original_post.NUM_CLASSES
        ):

            valid = (
                (class_map[lo:hi] == cls)
                & ~noise[lo:hi]
            ).astype(np.uint8)

            if not valid.any():
                continue

            dilated = cv2.dilate(
                valid, kernel
            )[y-lo:end-lo]

            fill = (
                noise[y:end]
                & dilated.astype(bool)
            )

            target[fill] = cls

    return result


def tag_harvest_zones(class_map, rows=256):

    H, W = class_map.shape

    cotton_idx = original_post.CLASS_TO_IDX.get(
        "cotton_ball", 3
    )

    window = original_post.HARVEST_WINDOW

    threshold = (
        original_post.HARVEST_COTTON_DENSITY
    )

    result = np.empty(
        (H, W), dtype=bool
    )

    for y, end, lo, hi in blocks(
        H, rows, window // 2
    ):

        cotton = (
            class_map[lo:hi] == cotton_idx
        ).astype(np.float32)

        density = cv2.blur(
            cotton, (window, window)
        )

        result[y:end] = (
            density[y-lo:end-lo] > threshold
        )

    return result


def postprocess(
    class_map,
    rgb,
    skip_morph=False,
    rows=256
):

    masks = compute_aux_masks(
        rgb, rows=rows
    )

    result = class_map.copy()

    for y, end, _, _ in blocks(
        result.shape[0], rows
    ):

        result[y:end][
            ~masks["valid_field"][y:end]
        ] = 0

    result = soften_problematic_pixels(
        result, rgb, masks, rows
    )

    del masks

    if not skip_morph:

        result = morphological_cleanup(
            result, rows
        )

    harvest = tag_harvest_zones(
        result, rows
    )

    return result, harvest


# BOUNDARY_COMPONENTS_V1

from src.preprocessing import boundary_components as boundary_cc
