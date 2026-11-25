
"""Boundary-only global connected component analysis."""

import cv2
import numpy as np
from pathlib import Path


def analyze(binary, strip_rows=256, *, work_dir):

    H, W = binary.shape

    if H <= 0 or W <= 0 or strip_rows <= 0:
        raise ValueError("Invalid dimensions")

    if min(H, strip_rows) * W >= 2**31:
        raise ValueError("Strip exceeds INT32 limit")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    parents = [0]
    sizes = [0]
    borders = [False]

    strips = []

    internal_largest = 0
    internal_count = 0

    previous = None

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
        borders[a] = borders[a] or borders[b]

    for y in range(0, H, strip_rows):

        end = min(y + strip_rows, H)

        block = np.ascontiguousarray(
            binary[y:end],
            dtype=np.uint8
        )

        n, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                block,
                connectivity=8
            )
        )

        local_areas = stats[
            :, cv2.CC_STAT_AREA
        ].astype(np.int64)

        local_areas[0] = 0

        local_borders = np.zeros(
            n, dtype=bool
        )

        local_borders[
            np.unique(labels[:, 0])
        ] = True

        local_borders[
            np.unique(labels[:, -1])
        ] = True

        if y == 0:

            local_borders[
                np.unique(labels[0])
            ] = True

        if end == H:

            local_borders[
                np.unique(labels[-1])
            ] = True

        local_borders[0] = False

        # Only components touching an internal
        # horizontal strip boundary need global IDs.
        boundary_parts = []

        if y > 0:
            boundary_parts.append(labels[0])

        if end < H:
            boundary_parts.append(labels[-1])

        if boundary_parts:

            boundary_ids = np.unique(
                np.concatenate(boundary_parts)
            )

            boundary_ids = boundary_ids[
                boundary_ids != 0
            ]

        else:

            boundary_ids = np.empty(
                0, dtype=np.int32
            )

        mapping = np.zeros(
            n, dtype=np.int64
        )

        for label in boundary_ids:

            label = int(label)

            gid = len(parents)

            mapping[label] = gid

            parents.append(gid)

            sizes.append(
                int(local_areas[label])
            )

            borders.append(
                bool(local_borders[label])
            )

        internal = (
            (mapping == 0)
            & (local_areas > 0)
        )

        internal_count += int(
            np.count_nonzero(internal)
        )

        if internal.any():

            internal_largest = max(
                internal_largest,
                int(local_areas[internal].max())
            )

        top = mapping[labels[0]]

        if previous is not None:

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

        previous = mapping[labels[-1]]

        prefix = work / f"strip_{y:08d}"

        label_path = str(prefix) + "_labels.npy"
        area_path = str(prefix) + "_areas.npy"
        border_path = str(prefix) + "_borders.npy"
        map_path = str(prefix) + "_mapping.npy"

        np.save(label_path, labels)
        np.save(area_path, local_areas)
        np.save(border_path, local_borders)
        np.save(map_path, mapping)

        strips.append((
            y, end,
            label_path,
            area_path,
            border_path,
            map_path
        ))

        del block, labels, stats, centroids
        del local_areas, local_borders
        del mapping, internal, top
        del boundary_parts, boundary_ids

    roots = np.asarray(
        [find(i) for i in range(len(parents))],
        dtype=np.int64
    )

    areas = np.asarray(
        sizes, dtype=np.int64
    )

    touches = np.asarray(
        borders, dtype=bool
    )

    non_roots = (
        roots != np.arange(len(roots))
    )

    areas[non_roots] = 0
    touches[non_roots] = False

    largest = max(
        internal_largest,
        int(areas.max())
    )

    count = (
        internal_count
        + int(np.count_nonzero(areas))
    )

    return {
        "shape": (H, W),
        "strips": strips,
        "roots": roots,
        "areas": areas,
        "borders": touches,
        "largest": largest,
        "count": count,
        "global_nodes": len(roots) - 1,
        "internal_components": internal_count
    }


def selected_mask(result, selector):

    output = np.zeros(
        result["shape"], dtype=bool
    )

    roots = result["roots"]
    areas = result["areas"]
    borders = result["borders"]

    global_selected = selector(
        areas, borders
    )

    for (
        y, end,
        label_path,
        area_path,
        border_path,
        map_path
    ) in result["strips"]:

        labels = np.load(
            label_path, mmap_mode="r"
        )

        local_areas = np.load(
            area_path, mmap_mode="r"
        )

        local_borders = np.load(
            border_path, mmap_mode="r"
        )

        mapping = np.load(
            map_path, mmap_mode="r"
        )

        lookup = np.asarray(
            selector(local_areas, local_borders),
            dtype=bool
        ).copy()

        crossing = mapping > 0

        lookup[crossing] = global_selected[
            roots[mapping[crossing]]
        ]

        lookup[0] = False

        output[y:end] = lookup[labels]

    return output
