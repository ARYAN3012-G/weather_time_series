# Architecture and Methodology Reference

This folder contains exact source-code extracts grouped
by technical purpose, together with Mermaid diagrams.

Claude should inspect the source implementations before
writing mathematical equations or architectural details.

In particular, extract:

- Exact vegetation-index equations.
- Feature normalization and clipping.
- Encoder attention heads and spatial reduction ratios.
- Patch sizes, strides and decoder configuration.
- Loss equations and class weighting.
- Optimizer and learning-rate schedule.
- Augmentation probabilities.
- Pseudo-label and synthetic sample weighting.
- Active postprocessing thresholds.
- Memory-safe component algorithms.

The diagrams are explanatory representations.
Check them against the active source before publication.

Mermaid files can be rendered into publication figures.

Do not invent parameter counts, missing metrics,
dataset statistics or unsupported implementation details.