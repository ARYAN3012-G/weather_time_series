"""
Report generation for orthomosaic predictions.

Outputs per orthomosaic:
    mask.png       Full-resolution color-coded segmentation mask
    overlay.jpg    50% alpha overlay on original with class legend (max 8192px wide)
    report.json    Per-class areas, drought decision, all metrics
    report.pdf     Detailed report: legend, area table, decision signals,
                   metric definitions glossary
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.gsd import gsd_mm_per_pixel, pixels_to_m2
from src.utils.constants import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, PALETTE

# ---------------------------------------------------------------------------
# Class descriptions for the report legend
# ---------------------------------------------------------------------------
CLASS_DESCRIPTIONS = {
    'background':   'No-data area outside the flight coverage boundary.',
    'soil':         'Bare or partially bare agricultural soil with no active crop cover.',
    'crop':         'Active standing crop canopy (primarily maize at vegetative stage).',
    'cotton_ball':  'Open cotton bolls visible — indicates harvest-ready or harvest stage.',
    'green_weed':   'Non-crop green vegetation (weeds) within or around the field.',
    'dry_weed':     'Dry or senescent weed vegetation; tan/brown in appearance.',
    'drought':      'Crop canopy showing stress symptoms: brown/red discolouration, '
                    'wilting, or die-back indicative of water deficit.',
}

# ---------------------------------------------------------------------------
# Area computation
# ---------------------------------------------------------------------------

def compute_areas(
    class_map: np.ndarray,
    harvest_mask: np.ndarray,
    altitude_m: float,
    gsd_override_mm=None,
) -> dict:
    """
    Compute per-class pixel counts and areas in m².

    Drought fraction is computed over the agricultural footprint only
    (crop + cotton_ball + drought + dry_weed) — not over soil or background.
    """
    gsd_mm  = (gsd_mm_per_pixel(altitude_m) if gsd_override_mm is None else float(gsd_override_mm))
    px_m2   = (gsd_mm / 1000.0) ** 2

    areas: dict[str, dict] = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count    = int((class_map == cls_idx).sum())
        area_m2  = count * px_m2
        areas[cls_name] = {'pixels': count, 'area_m2': round(area_m2, 2)}

    total_px       = int((class_map < NUM_CLASSES).sum()) - int((class_map == 0).sum())
    valid_field_px = total_px

    agri_classes   = ['crop', 'cotton_ball', 'drought', 'dry_weed']
    agri_px        = sum(areas[c]['pixels'] for c in agri_classes)
    drought_px     = areas['drought']['pixels']

    drought_fraction = drought_px / max(agri_px, 1)

    non_harvest_drought = int(
        ((class_map == CLASS_TO_IDX.get('drought', 6)) & ~harvest_mask).sum()
    )
    non_harvest_agri = 0

    for cls_name in (
        "crop", "cotton_ball", "drought", "dry_weed"
    ):
        cls_idx = CLASS_TO_IDX[cls_name]

        non_harvest_agri += int(
            np.count_nonzero(
                (class_map == cls_idx) & ~harvest_mask
            )
        )

    non_harvest_agri = max(non_harvest_agri, 1)
    non_harvest_drought_frac = non_harvest_drought / non_harvest_agri

    return {
        'gsd_mm_per_pixel':               round(gsd_mm, 3),
        'pixel_area_m2':                  px_m2,
        'altitude_m':                     altitude_m,
        'valid_field_pixels':             valid_field_px,
        'valid_field_area_m2':            round(valid_field_px * px_m2, 1),
        'agricultural_pixels':            agri_px,
        'agricultural_area_m2':           round(agri_px * px_m2, 1),
        'drought_fraction':               drought_fraction,
        'non_harvest_drought_fraction':   non_harvest_drought_frac,
        'per_class':                      areas,
    }


# ---------------------------------------------------------------------------
# Drought decision
# ---------------------------------------------------------------------------

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
    drought_idx    = CLASS_TO_IDX.get('drought', 6)
    drought_binary = (class_map == drought_idx).astype(np.uint8)
    n_drought_px   = int(drought_binary.sum())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        drought_binary, connectivity=8
    )
    if num_labels > 1:
        cc_sizes   = stats[1:, cv2.CC_STAT_AREA]
        largest_cc = int(cc_sizes.max()) if len(cc_sizes) > 0 else 0
        cc_share   = largest_cc / max(n_drought_px, 1)
    else:
        largest_cc, cc_share = 0, 0.0

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


# ---------------------------------------------------------------------------
# Visual outputs
# ---------------------------------------------------------------------------

def make_color_mask(class_map: np.ndarray) -> np.ndarray:
    """Convert class-index map to (H, W, 3) uint8 RGB color mask."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(PALETTE):
        rgb[class_map == cls_idx] = color
    return rgb


def _draw_legend_panel(height: int, width: int = 220) -> np.ndarray:
    """Build a white legend panel with colour swatches for every class."""
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    y     = 25
    cv2.putText(panel, 'LEGEND', (10, y), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 8
    cv2.line(panel, (10, y), (width - 10, y), (180, 180, 180), 1)
    y += 15
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_name == 'background':
            continue
        color_rgb = PALETTE[cls_idx]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.rectangle(panel, (10, y - 12), (28, y + 4), color_bgr, -1)
        cv2.rectangle(panel, (10, y - 12), (28, y + 4), (100, 100, 100), 1)
        cv2.putText(panel, cls_name, (34, y), font, 0.42,
                    (30, 30, 30), 1, cv2.LINE_AA)
        y += 22
    return panel


def save_mask_png(class_map: np.ndarray, path: Path) -> None:
    """Save segmentation as indexed PNG with original class colors."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(class_map.astype(np.uint8, copy=False), mode='P')

    palette = []
    for color in PALETTE:
        palette.extend([int(v) for v in color[:3]])

    palette.extend([0] * (768 - len(palette)))
    image.putpalette(palette)
    image.save(str(path), format='PNG')


def save_overlay_jpg(
    img_rgb: np.ndarray,
    class_map: np.ndarray,
    path: Path,
    max_width: int = 8192,
) -> None:
    """
    Save semi-transparent overlay with legend panel on the right.
    Resizes to max_width if the image is very large.
    """
    H, W = img_rgb.shape[:2]
    if W > max_width:
        scale  = max_width / W
        img_s  = cv2.resize(img_rgb,   (max_width, int(H * scale)),
                             interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(class_map, (max_width, int(H * scale)),
                             interpolation=cv2.INTER_NEAREST)
    else:
        img_s, mask_s = img_rgb, class_map

    color_s = make_color_mask(mask_s)
    overlay = (0.5 * img_s.astype(np.float32) +
               0.5 * color_s.astype(np.float32)).astype(np.uint8)

    # Stat bar at bottom
    names  = CLASS_NAMES
    total  = max((mask_s > 0).sum(), 1)
    stats  = ' | '.join(
        f'{names[i]}={int((mask_s==i).sum())/total*100:.1f}%'
        for i in range(NUM_CLASSES)
        if (mask_s == i).sum() > 0 and names[i] != 'background'
    )
    bar_h  = 28
    bar    = np.full((bar_h, overlay.shape[1], 3), 240, dtype=np.uint8)
    cv2.putText(bar, stats, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (40, 40, 40), 1, cv2.LINE_AA)
    overlay = np.vstack([overlay, bar])

    # Legend panel on the right
    legend    = _draw_legend_panel(overlay.shape[0])
    composite = np.hstack([overlay, legend])

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def save_json_report(
    area_stats: dict,
    decision: dict,
    img_path: str,
    out_path: Path,
) -> None:
    report = {
        'source_image':     img_path,
        'area_stats':       area_stats,
        'drought_decision': decision,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def save_pdf_report(
    img_rgb: np.ndarray,
    class_map: np.ndarray,
    area_stats: dict,
    decision: dict,
    img_name: str,
    out_path: Path,
) -> None:
    """
    Detailed PDF report:
        - Drought decision banner
        - Original thumbnail | colour mask | class legend with descriptions
        - Per-class area breakdown table
        - Decision signals table with metric values and thresholds
        - Full glossary: definitions of every metric used
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle,
            Paragraph, Spacer, Image as RLImage, HRFlowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import tempfile, os
        from PIL import Image as PILImage
    except ImportError:
        print('  [WARNING] reportlab not available — skipping PDF')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc    = SimpleDocTemplate(str(out_path), pagesize=A4,
                               leftMargin=1.8*cm, rightMargin=1.8*cm,
                               topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('t',  parent=styles['Title'],    fontSize=14, spaceAfter=4)
    h2_style    = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=10, spaceBefore=8, spaceAfter=3)
    h3_style    = ParagraphStyle('h3', parent=styles['Heading3'], fontSize=9,  spaceBefore=5, spaceAfter=2)
    body_style  = ParagraphStyle('b',  parent=styles['Normal'],   fontSize=8,  leading=11)
    small_style = ParagraphStyle('s',  parent=styles['Normal'],   fontSize=7,  leading=10,
                                 textColor=colors.HexColor('#333333'))

    is_drought = decision['drought']
    story      = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Banner ─────────────────────────────────────────────────────────────
        story.append(Paragraph(f'Drought Analysis Report — {img_name}', title_style))
        call_col  = '#CC0000' if is_drought else '#007700'
        call_text = '⚠  VISUAL DROUGHT-STRESS FLAG' if is_drought else '✓  No field-level stress flag'
        story.append(Paragraph(
            f'<b><font color="{call_col}" size="12">{call_text}</font></b>'
            f'&nbsp;&nbsp;<font size="8" color="grey">'
            f'Heuristic stress score: {decision["confidence"]:.0%}</font>',
            ParagraphStyle('call', parent=styles['Normal'],
                           fontSize=12, spaceAfter=6)))
        story.append(HRFlowable(width='100%', thickness=1,
                                color=colors.HexColor(call_col)))
        story.append(Spacer(1, 0.3*cm))

        # ── Images + Legend ────────────────────────────────────────────────────
        story.append(Paragraph('Field Overview', h2_style))

        H, W   = img_rgb.shape[:2]
        scale  = min(1.0, 300 / max(H, W))
        th_w, th_h = int(W * scale), int(H * scale)
        thumb  = cv2.resize(img_rgb,            (th_w, th_h), interpolation=cv2.INTER_AREA)
        small_labels = cv2.resize(
            class_map, (th_w, th_h),
            interpolation=cv2.INTER_NEAREST
        )
        c_res = make_color_mask(small_labels)

        tp = os.path.join(tmpdir, 'thumb.jpg')
        mp = os.path.join(tmpdir, 'mask.png')
        cv2.imwrite(tp, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        PILImage.fromarray(c_res).save(mp)

        img_w = 6.0*cm
        img_h = img_w * th_h / th_w

        # Legend table
        leg_rows = [[Paragraph('<b>Class</b>', small_style),
                     Paragraph('<b>Description</b>', small_style)]]
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            if cls_name == 'background':
                continue
            rgb = PALETTE[cls_idx]
            sw_color = colors.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            swatch   = Table([['']], colWidths=[0.4*cm], rowHeights=[0.35*cm])
            swatch.setStyle(TableStyle([
                ('BACKGROUND', (0,0),(0,0), sw_color),
                ('GRID',       (0,0),(0,0), 0.5, colors.grey),
            ]))
            leg_rows.append([
                Table([[swatch, Paragraph(f' {cls_name}', small_style)]],
                      colWidths=[0.5*cm, 2.3*cm]),
                Paragraph(CLASS_DESCRIPTIONS.get(cls_name, ''), small_style),
            ])
        leg_tbl = Table(leg_rows, colWidths=[2.9*cm, 7.0*cm])
        leg_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#444444')),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('GRID',  (0,0),(-1,-1), 0.3, colors.lightgrey),
            ('FONTSIZE',      (0,0),(-1,-1), 7),
            ('TOPPADDING',    (0,0),(-1,-1), 2),
            ('BOTTOMPADDING', (0,0),(-1,-1), 2),
        ]))

        img_row = [[RLImage(tp, width=img_w, height=img_h),
                    RLImage(mp, width=img_w, height=img_h),
                    leg_tbl]]
        img_tbl = Table(img_row, colWidths=[img_w+0.3*cm, img_w+0.3*cm, 10.2*cm])
        img_tbl.setStyle(TableStyle([
            ('ALIGN',  (0,0),(-1,-1), 'CENTER'),
            ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ]))
        story.append(img_tbl)
        story.append(Paragraph(
            '<i>Left: original orthomosaic &nbsp; '
            'Centre: segmentation mask &nbsp; '
            'Right: class legend with descriptions</i>', small_style))
        story.append(Spacer(1, 0.3*cm))

        # ── Area breakdown ─────────────────────────────────────────────────────
        story.append(Paragraph('Area Breakdown', h2_style))
        story.append(Paragraph(
            f'Area estimate (physical scale not independently verified): '
            f'<b>{"Review required" if area_stats.get("area_requires_review") else "Estimated"}</b><br/>'
            f'GSD: <b>{area_stats["gsd_mm_per_pixel"]} mm/px</b>  |  '
            f'Altitude: <b>{str(area_stats["altitude_m"]) + " m" if area_stats["altitude_m"] is not None else "Unknown"}</b>  |  '
            f'Field area: <b>{area_stats["valid_field_area_m2"]:,.0f} m²</b>  |  '
            f'Agricultural area: <b>{area_stats["agricultural_area_m2"]:,.0f} m²</b>',
            body_style))
        story.append(Spacer(1, 0.1*cm))

        valid  = max(area_stats['valid_field_area_m2'], 1)
        agri   = max(area_stats['agricultural_area_m2'], 1)
        a_data = [[Paragraph('<b>Class</b>',      small_style),
                   Paragraph('<b>Pixels</b>',     small_style),
                   Paragraph('<b>Area (m²)</b>',  small_style),
                   Paragraph('<b>% of field</b>', small_style),
                   Paragraph('<b>% of agri</b>',  small_style)]]
        for cls_name in CLASS_NAMES:
            if cls_name == 'background':
                continue
            d  = area_stats['per_class'].get(cls_name, {})
            a  = d.get('area_m2', 0)
            px = d.get('pixels', 0)
            pf = a / valid * 100
            pa = f'{a/agri*100:.1f}%' if cls_name in \
                 ['crop','cotton_ball','drought','dry_weed'] else '—'
            a_data.append([
                Paragraph(cls_name, small_style),
                Paragraph(f'{px:,}',     small_style),
                Paragraph(f'{a:,.1f}',   small_style),
                Paragraph(f'{pf:.1f}%',  small_style),
                Paragraph(pa,            small_style),
            ])
        at = Table(a_data, colWidths=[2.8*cm, 3.0*cm, 3.0*cm, 2.5*cm, 2.5*cm])
        at.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#444444')),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('GRID',  (0,0),(-1,-1), 0.3, colors.lightgrey),
            ('ROWBACKGROUNDS', (0,1),(-1,-1),
             [colors.white, colors.HexColor('#f5f5f5')]),
            ('FONTSIZE',      (0,0),(-1,-1), 7),
            ('TOPPADDING',    (0,0),(-1,-1), 2),
            ('BOTTOMPADDING', (0,0),(-1,-1), 2),
        ]))
        story.append(at)
        story.append(Spacer(1, 0.3*cm))

        # ── Decision signals ───────────────────────────────────────────────────
        story.append(Paragraph('Drought Decision Signals', h2_style))
        s_data = [[Paragraph('<b>Signal</b>',    small_style),
                   Paragraph('<b>Metric</b>',     small_style),
                   Paragraph('<b>Value</b>',      small_style),
                   Paragraph('<b>Threshold</b>',  small_style),
                   Paragraph('<b>Triggered</b>',  small_style)]]
        rows = [
            ('Signal 1', 'Drought fraction (all agri area)',
             f'{decision["drought_fraction"]:.6%}',
             f'> {decision["tau_fraction"]:.0%}',
             decision['signal_1_fraction']),
            ('Signal 2', 'Non-harvest drought fraction',
             f'{decision["non_harvest_drought_fraction"]:.6%}',
             f'> {decision["tau_non_harvest"]:.0%}',
             decision['signal_2_non_harvest']),
            ('Signal 3', 'Largest CC share (spatial coherence)',
             f'{decision["largest_cc_share"]:.1%}',
             f'> {decision["tau_cc_share"]:.0%}',
             decision['signal_3_coherent']),
        ]
        for r in rows:
            tick = '✓  Yes' if r[4] else '✗  No'
            col  = 'green'  if r[4] else 'red'
            s_data.append([
                Paragraph(r[0], small_style),
                Paragraph(r[1], small_style),
                Paragraph(f'<b>{r[2]}</b>', small_style),
                Paragraph(r[3], small_style),
                Paragraph(f'<font color="{col}"><b>{tick}</b></font>', small_style),
            ])
        st = Table(s_data, colWidths=[2.0*cm, 5.8*cm, 2.2*cm, 2.3*cm, 1.8*cm])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#444444')),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('GRID',  (0,0),(-1,-1), 0.3, colors.lightgrey),
            ('ROWBACKGROUNDS', (0,1),(-1,-1),
             [colors.white, colors.HexColor('#f5f5f5')]),
            ('FONTSIZE',      (0,0),(-1,-1), 7),
            ('TOPPADDING',    (0,0),(-1,-1), 3),
            ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ]))
        story.append(st)

        rule = ('Signal 1 alone (drought fraction exceeds threshold).'
                if decision['signal_1_fraction'] else
                'Signal 2 AND Signal 3 (moderate fraction + spatial coherence).')
        story.append(Paragraph(
            f'<i>Decision rule applied: {rule}</i>', small_style))
        story.append(Spacer(1, 0.3*cm))

        # ── Glossary ───────────────────────────────────────────────────────────
        story.append(HRFlowable(width='100%', thickness=0.5,
                                color=colors.lightgrey))
        story.append(Paragraph('Metric Definitions', h3_style))

        glossary = [
            ('<b>GSD (Ground Sampling Distance)</b>',
             'Real-world size of one image pixel. '
             'Area uses the reported orthomosaic pixel resolution when available. '
             'Used to convert pixel counts to m².'),
            ('<b>Agricultural Area</b>',
             'Total pixels of crop + cotton_ball + drought + dry_weed. '
             'Excludes background and bare soil. Drought fractions are computed '
             'over this area so soil-dominated fields do not falsely dilute the signal.'),
            ('<b>Drought Fraction (Signal 1)</b>',
             f'Proportion of agricultural area classified as drought. '
             f'Threshold: {decision["tau_fraction"]:.0%}. '
             'Exceeding this threshold alone is sufficient to raise a visual stress flag '
             'because it means significant crop area is visibly stressed.'),
            ('<b>Non-Harvest Drought Fraction (Signal 2)</b>',
             'Drought fraction after removing pixels inside harvest zones '
             '(regions where cotton_ball density is high). '
             'In cotton fields at harvest, senescent leaves mimic drought colour; '
             f'this metric discounts those zones. Threshold: {decision["tau_non_harvest"]:.0%}.'),
            ('<b>CC (Connected Component)</b>',
             'A group of spatially adjacent pixels all classified as drought. '
             'Pixels are connected if they share an edge or corner (8-connectivity). '
             'Many small CCs = scattered noise. One large CC = coherent stress patch.'),
            ('<b>Largest CC Share (Signal 3)</b>',
             f'Fraction of all drought pixels in the single largest connected drought region. '
             f'Threshold: {decision["tau_cc_share"]:.0%}. '
             'Above threshold = spatially coherent predicted stress. '
             'Below threshold = scattered pixels (likely misclassification noise). '
             'Signal 3 is used as a coherence gate alongside Signal 2 but NOT Signal 1.'),
            ('<b>Heuristic Stress Score</b>',
             'Composite index (0–100%) computed as: '
             '0.5 × (drought_frac / threshold) + '
             '0.3 × (non_harvest_frac / threshold) + '
             '0.2 × coherence_signal. '
             'Capped at 100%. This score is not a calibrated probability or confirmed drought diagnosis.'),
        ]
        for term, defn in glossary:
            story.append(Paragraph(f'{term}: {defn}', body_style))
            story.append(Spacer(1, 0.07*cm))

        doc.build(story)
