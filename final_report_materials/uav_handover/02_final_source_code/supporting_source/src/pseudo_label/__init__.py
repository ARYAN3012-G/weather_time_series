"""Pseudo-labeling of unlabeled UAV captures using the patch classifier."""
from .label import label
from .postprocess import apply_crf, postprocess_probmap
from .sliding_window import SlidingWindowInference

__all__ = [
    "label",
    "apply_crf",
    "postprocess_probmap",
    "SlidingWindowInference",
]
