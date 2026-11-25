
"""Memory-efficient drought reporting."""

import numpy as np
import tempfile
import importlib

reference = importlib.import_module(
    "src.inference.report"
)

CLASS_TO_IDX = reference.CLASS_TO_IDX

from src.preprocessing.disk_components import (
    strip_components
)

def drought_decision(
    area_stats: dict,
    harvest_mask: np.ndarray,
    class_map: np.ndarray,
    tau_fraction: float    = 0.10,
    tau_non_harvest: float = 0.08,
    tau_cc_share: float    = 0.20,
) -> dict:
    """
    Three-signal drought decision.

    Signal 1 — Drought fraction > tau_fraction (10%)
        Fraction of the agricultural area (crop+cotton+drought+dry_weed)
        classified as drought.

    Signal 2 — Non-harvest drought fraction > tau_non_harvest (8%)
        Same as Signal 1 but excludes harvest zones (high cotton_ball density)
        to avoid false positives from senescent cotton leaves around open bolls.

    Signal 3 — Largest Connected Component (CC) share > tau_cc_share (20%)
        Fraction of ALL drought pixels belonging to the single largest
        spatially connected drought region. High = coherent stress zone.
        Low = scattered noise / misclassification.

    Decision: DROUGHT if Signal 1 alone (high fraction) OR (Signal 2 AND Signal 3).
    """

    drought_idx = CLASS_TO_IDX.get("drought", 6)

    H, W = class_map.shape

    drought_binary = np.empty(
        (H, W), dtype=np.uint8
    )

    n_drought_px = 0

    for y in range(0, H, 256):

        end = min(y + 256, H)

        block = (
            class_map[y:end] == drought_idx
        )

        drought_binary[y:end] = block

        n_drought_px += int(
            np.count_nonzero(block)
        )

    largest_cc = 0
    cc_share = 0.0

    if n_drought_px > 0:

        with tempfile.TemporaryDirectory(
            prefix="drought_decision_",
            dir="/content"
        ) as work:

            result = boundary_cc.analyze(
                drought_binary,
                strip_rows=256,
                work_dir=work
            )

            largest_cc = int(result["largest"])

            cc_share = (
                largest_cc / n_drought_px
            )

            del result

    del drought_binary

    df   = area_stats['drought_fraction']
    nhdf = area_stats['non_harvest_drought_fraction']
    sig1 = df   > tau_fraction
    sig2 = nhdf > tau_non_harvest
    sig3 = cc_share > tau_cc_share

    # Updated rule: Signal 1 alone is sufficient (handles distributed row drought)
    is_drought = sig1 or (sig2 and sig3)
    confidence = round(min(1.0,
        (df   / max(tau_fraction,    1e-6)) * 0.5 +
        (nhdf / max(tau_non_harvest, 1e-6)) * 0.3 +
        float(sig3) * 0.2
    ), 3)

    return {
        'drought':                         is_drought,
        'confidence':                      confidence,
        'stress_score': confidence,
        'score_is_calibrated_probability': False,
        'decision_type': 'visual_stress_heuristic',
        'field_confirmation_required': True,
        'drought_fraction':                df,
        'non_harvest_drought_fraction':    nhdf,
        'largest_cc_share':                round(cc_share, 4),
        'signal_1_fraction':               sig1,
        'signal_2_non_harvest':            sig2,
        'signal_3_coherent':               sig3,
        'tau_fraction':                    tau_fraction,
        'tau_non_harvest':                 tau_non_harvest,
        'tau_cc_share':                    tau_cc_share,
    }


# BOUNDARY_COMPONENTS_V1

from src.preprocessing import boundary_components as boundary_cc
