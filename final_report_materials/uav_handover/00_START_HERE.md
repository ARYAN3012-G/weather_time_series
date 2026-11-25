# UAV MODULE 3 - COMPLETE RESEARCH HANDOVER

This folder documents Module 3 of an integrated research project.

Modules 1 and 2 are already documented separately.

## Recommended reading order

1. 01_COMPLETE_UAV_KNOWLEDGE.md
   Complete project knowledge, methodology and results.

2. 13_architecture_and_methodology/
   Exact implementation references and six technical diagrams.

3. 12_custom_annotation_tool/
   Custom human-in-the-loop annotation source and methodology.

4. 02_final_source_code/
   Final V11 implementation and supporting source code.

5. 03_dataset_and_training/
   Actual training history, checkpoint metadata and pending
   independent evaluation dataset records.

6. 04_evaluation_results/
   Model comparisons, calibration and validation evidence.

7. 05_inference_optimization/
   Stitching, memory optimization, runtime and equivalence tests.

8. 06_field_results/
   Field-wise and class-wise results and individual reports.

9. 07_figures/
   Research graphs and representative segmentation overlays.

10. 10_technical_reference/
    Consolidated technical source reference.

11. 11_experimental_history/
    Historical configurations and dataset provenance.

## Final selected implementation

SegFormer-B2 V11 with nine-channel input, seven semantic
classes, class-specific logit adjustment, Hann-weighted
rolling stitching and memory-safe postprocessing.

## Key recorded results

Raw validation mIoU: 61.46 percent.
Adjusted validation mIoU: 65.11 percent.
Adjusted validation pixel accuracy: 82.68 percent.
Final orthomosaics processed: 14.
Original final output files: 56.

These are validation metrics and field predictions,
not independent test results.

## Important research restrictions

- Do not invent missing training metrics.
- Do not confuse historical versions with final V11.
- Do not claim independent testing is complete.
- Do not interpret visual drought flags as confirmed drought.
- Do not interpret heuristic scores as calibrated probabilities.
- Do not claim physical area estimates are surveyed measurements.
- Distinguish UAV photograph stitching from segmentation
  prediction stitching.
- Do not claim the custom annotation tool is CVAT.
- Verify exact equations and parameters from source code.
- Preserve the existing report structure of Modules 1 and 2.
- Do not invent cross-module integration mechanisms.
- Do not invent research citations.

Raw datasets, full-resolution segmentation masks and
model checkpoints are intentionally excluded from this
documentation handover.

Actual source code and recorded evidence take precedence
over explanatory summaries when discrepancies arise.