# Chat summary – start here

Location: `weather_module/chat_summary/` (main folder). Commands below are run from `weather_module/final_report_materials/`.

Saved on 25 Nov 2026, at the end of the long chat that built the final report and the end-semester slides.
These notes keep everything important from that chat in a few files.

## The files in this folder

| File | What is inside |
|---|---|
| `00_README.md` | This page: where the final files are, how to rebuild them, rules to remember |
| `01_modules_and_results.md` | Each module (weather, satellite, rabi, UAV): data, method, results, and why |
| `02_viva_explanations.md` | Simple explanations of every question asked in the chat (AUC, persistence, RF, …) with ready-to-say viva lines |
| `03_slides_and_report_changes.md` | What was changed in the slides and the report, and the design decisions |
| `04_references_checked.md` | Literature check: which papers were verified, which were wrong and fixed, which old papers were kept and why |
| `tools/check_all.py` | Checks every slide for text running into the footer and makes a contact sheet |
| `tools/make_zip.py` | Rebuilds the Overleaf zip with only the figures the two .tex files use |

## Final outputs (all in `final_report_materials/`)

| File | What it is |
|---|---|
| `Final_Presentation.pdf` | End-semester slides, 29 slides, "clean card" theme |
| `Final_Report_v2.pdf` | Final report, 18 pages, module-wise |
| `Final_Report_v2_and_Slides_Overleaf.zip` | `main_v2.tex` + `presentation_final.tex` + 26 figures, ready to upload to Overleaf |
| `latex/presentation_final.tex` | Slide source |
| `latex/main_v2.tex` | Report source |
| `latex/presentation_navy_backup.tex` | Old navy-theme slides (backup) |
| `latex/build/presentation_before_rabi.tex` | Slides before the rabi section was simplified (backup) |

Diagram scripts (in `final_report_materials/`):
- `make_diagrams_v2.py`: report diagrams.
- `make_*_slide.py`: wide slide diagrams (system, weather, satellite, rabi, UAV pipeline, UAV model).
- `make_uav_steps.py`: UAV step images.

## How to rebuild the PDFs on the laptop

LaTeX = MiKTeX, installed for the user only: `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`.
It installs missing packages automatically.

```bash
cd final_report_materials/latex
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -output-directory=build presentation_final.tex   # run twice
pdflatex -interaction=nonstopmode -output-directory=build main_v2.tex              # run 2-3 times
```

If `build/presentation_final.pdf` is open in a viewer, Windows locks it. Then build with
`-jobname=pres_check` and copy `build/pres_check.pdf` to `../Final_Presentation.pdf`.

After any slide change, run the checks:
```bash
python -I ../chat_summary/tools/check_all.py latex/build/pres_check.pdf sheet.png
```
The only flag that is expected is slide 1: the title slide has no footer, so it is a false alarm.

Then rebuild the zip:
```bash
python -I ../chat_summary/tools/make_zip.py
```

## Rules to remember (from the chat)

- **Security:** the CDS API token is in `C:\Users\aryan\.cdsapirc`. Never paste or commit it.
- **Git:** never commit or push without explicit permission. Nothing from this chat was committed. `final_report_materials/` and `rabi_module_v2/` are not in git.
- **Honesty:** don't overclaim.
  - Validation is not independent testing.
  - RGB drone images show visual stress, not its cause.
  - Don't invent citations or numbers.
- **Training:** heavy training runs on the lab desktop only. The laptop is for light checks.
- **Zenodo / DOI:** only after the professors accept the work.
- **Slides:** no phone numbers or emails from the professor's old deck.
- **Writing tips:**
  - Bash heredocs break LaTeX backslashes, so write edit scripts as `.py` files and run them with `python -I`.
  - In Python, put LaTeX strings that contain `''` (closing quotes) inside triple quotes `r"""..."""`. Otherwise the quotes get lost.
- **Beamer tips:**
  - A font size set before `columns` is reset inside each `\column`, so set it inside the column.
  - Lists (`itemize`/`enumerate`) reset the font. Use `\setbeamerfont{itemize/enumerate body}{...}` inside the frame.
