import sys
import pymupdf
from PIL import Image

pdf, out = sys.argv[1], sys.argv[2]
d = pymupdf.open(pdf)
bad = []
for n, pg in enumerate(d, 1):
    W, H = pg.rect.width, pg.rect.height
    foot = H - 0.06 * H  # footer band top
    for x0, y0, x1, y1, txt, *_ in pg.get_text('blocks'):
        t = txt.strip().replace('\n', ' ')
        if 'IIITDM Kurnool' in t or t.replace(' ', '').replace('/', '').isdigit():
            continue
        if y1 > foot or x1 > W - 2 or x0 < 2:
            bad.append((n, round(y1, 1), round(foot, 1), t[:60]))
    for x0, y0, x1, y1 in [im['bbox'] for im in pg.get_image_info()]:
        if y1 > foot + 1 or x1 > W + 1:
            bad.append((n, 'img', round(y1, 1), round(x1, 1)))
print('issues:', bad if bad else 'none')
imgs = [Image.frombytes('RGB', (p.width, p.height), p.samples) for p in (pg.get_pixmap(dpi=45) for pg in d)]
w, h = imgs[0].size
cols = 4
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * h), 'white')
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(out)
