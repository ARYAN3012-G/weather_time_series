"""
Ground Sample Distance (GSD) calculation.

GSD = pixel_size × altitude / focal_length

Verified against the user's Metashape report:
    pixel_size = 1.31 μm, focal = 6.72 mm, altitude = 13.8 m
    → GSD = 1.31e-3 × 13800 / 6.72 = 2.690 mm/pix  ✓ (report says 2.69 mm/pix)

This module is the single source of truth for converting pixels to m² in reports.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics. Defaults are from the user's Metashape report (DJI FC8482)."""
    pixel_size_um: float = 1.31
    focal_length_mm: float = 6.72
    native_width_px: int = 8064
    native_height_px: int = 4536


def gsd_mm_per_pixel(altitude_m: float, cam: CameraIntrinsics | None = None) -> float:
    """
    Compute Ground Sample Distance in mm/pixel for a given flight altitude.

    Args:
        altitude_m: Flight altitude above ground level, in meters.
        cam: Camera intrinsics. Defaults to the FC8482 from the report.

    Returns:
        GSD in mm/pixel.
    """
    if cam is None:
        cam = CameraIntrinsics()
    if altitude_m <= 0:
        raise ValueError(f"altitude_m must be positive; got {altitude_m}")
    # pixel_size [mm] × altitude [mm] / focal_length [mm]  →  [mm/pix]
    pixel_size_mm = cam.pixel_size_um * 1e-3
    altitude_mm = altitude_m * 1e3
    return pixel_size_mm * altitude_mm / cam.focal_length_mm


def pixel_area_m2(altitude_m: float, cam: CameraIntrinsics | None = None) -> float:
    """Area covered by one pixel on the ground, in m²."""
    gsd_mm = gsd_mm_per_pixel(altitude_m, cam)
    gsd_m = gsd_mm * 1e-3
    return gsd_m * gsd_m


def pixels_to_m2(num_pixels: int, altitude_m: float, cam: CameraIntrinsics | None = None) -> float:
    """Convert a pixel count to ground area in m²."""
    return num_pixels * pixel_area_m2(altitude_m, cam)


def pixels_to_hectares(num_pixels: int, altitude_m: float, cam: CameraIntrinsics | None = None) -> float:
    """Convert a pixel count to ground area in hectares."""
    return pixels_to_m2(num_pixels, altitude_m, cam) / 10_000.0


# Sanity-check on import (catches regressions)
def _self_test() -> None:
    cam = CameraIntrinsics()
    # The Metashape report says GSD = 2.69 mm/pix at altitude 13.8m
    gsd = gsd_mm_per_pixel(13.8, cam)
    assert abs(gsd - 2.69) < 0.01, f"GSD self-test failed: got {gsd:.3f}, expected ~2.69"


_self_test()
