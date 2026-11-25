import re
import zipfile
from pathlib import Path

root = Path(r'c:\Users\aryan\OneDrive\Desktop\weather_module\final_report_materials')
lat = root / 'latex'
texs = ['main_v2.tex', 'presentation_final.tex']
figs = set()
for t in texs:
    s = (lat / t).read_text(encoding='utf-8')
    s = '\n'.join(l.split('%')[0] if not l.lstrip().startswith('%') else '' for l in s.splitlines())
    for m in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s):
        figs.add(m)
missing = []
out = root / 'Final_Report_v2_and_Slides_Overleaf.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for t in texs:
        z.write(lat / t, t)
    for f in sorted(figs):
        p = lat / f
        if not p.exists():
            cands = [lat / (f + e) for e in ('.png', '.jpg', '.jpeg', '.pdf')]
            cands = [c for c in cands if c.exists()]
            if not cands:
                missing.append(f)
                continue
            p = cands[0]
        z.write(p, p.relative_to(lat).as_posix())
print('figures:', len(figs), 'missing:', missing)
print(len(zipfile.ZipFile(out).namelist()), 'files in zip')
