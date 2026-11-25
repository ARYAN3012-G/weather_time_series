
"""Strip-based global 8-connected component analysis."""

import cv2
import numpy as np


def strip_components(binary, strip_rows=256):
    """
    Label components in strips and merge across boundaries.

    Returns:
        strips: local label arrays and global offsets
        roots: global component root mapping
        areas: int64 global component areas
        borders: global border-touching flags
    """
    H, W = binary.shape

    if H == 0 or W == 0 or strip_rows <= 0:
        raise ValueError("Invalid image or strip size")

    if min(H, strip_rows) * W >= 2**31:
        raise ValueError("Strip exceeds int32 pixel limit")

    parents = [0]
    sizes = [0]
    touches = [False]
    strips = []

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)

        if a == b:
            return

        if sizes[a] < sizes[b]:
            a, b = b, a

        parents[b] = a
        sizes[a] += sizes[b]
        touches[a] = touches[a] or touches[b]

    previous = None

    for y in range(0, H, strip_rows):

        end = min(y + strip_rows, H)

        block = np.ascontiguousarray(
            binary[y:end], dtype=np.uint8
        )

        n, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                block, connectivity=8
            )
        )

        offset = len(parents) - 1

        parents.extend(
            range(offset + 1, offset + n)
        )

        sizes.extend(
            stats[1:, cv2.CC_STAT_AREA]
            .astype(np.int64).tolist()
        )

        touches.extend([False] * (n - 1))

        edges = [
            labels[:, 0],
            labels[:, -1]
        ]

        if y == 0:
            edges.append(labels[0])

        if end == H:
            edges.append(labels[-1])

        for edge in edges:
            for label in np.unique(edge):
                if label:
                    touches[offset + int(label)] = True

        top = labels[0].astype(np.int64)

        top = np.where(
            top > 0, top + offset, 0
        )

        if previous is not None:

            # Vertical and diagonal connections.
            for shift in (-1, 0, 1):

                if shift == -1:
                    a, b = previous[1:], top[:-1]
                elif shift == 1:
                    a, b = previous[:-1], top[1:]
                else:
                    a, b = previous, top

                valid = (a > 0) & (b > 0)

                if not valid.any():
                    continue

                pairs = np.unique(
                    np.column_stack(
                        (a[valid], b[valid])
                    ),
                    axis=0
                )

                for left, right in pairs:
                    union(int(left), int(right))

        bottom = labels[-1].astype(np.int64)

        previous = np.where(
            bottom > 0, bottom + offset, 0
        )

        strips.append((y, end, labels, offset))

    roots = np.asarray(
        [find(i) for i in range(len(parents))],
        dtype=np.int64
    )

    areas = np.asarray(sizes, dtype=np.int64)
    borders = np.asarray(touches, dtype=bool)

    non_roots = roots != np.arange(len(roots))

    areas[non_roots] = 0
    borders[non_roots] = False

    return strips, roots, areas, borders


def component_mask(
    shape, strips, roots, selected
):
    """Reconstruct selected global components."""

    output = np.zeros(shape, dtype=bool)

    for y, end, labels, offset in strips:

        lookup = np.zeros(
            int(labels.max()) + 1,
            dtype=bool
        )

        ids = np.arange(
            1, len(lookup), dtype=np.int64
        )

        lookup[1:] = selected[
            roots[ids + offset]
        ]

        output[y:end] = lookup[labels]

    return output
