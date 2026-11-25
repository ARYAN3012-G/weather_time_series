"""Build report additions from saved drought_v1 results, without fitting models.

Run after update_rabi_events.py; this is the final report integration step.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LATEX = HERE / 'latex'
FIG = LATEX / 'figures'
RES = ROOT / 'rabi_module/results/drought_v1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def table(caption, label, headings, rows, spec):
    return '\n'.join([r'\begin{table}[H]', r'\centering\small',
        r'\caption{'+caption+'}', r'\label{'+label+'}',
        r'\begin{tabularx}{\textwidth}{@{}'+spec+'@{}}', r'\toprule',
        ' & '.join(headings)+r' \\', r'\midrule'] +
        [' & '.join(map(str, row))+r' \\' for row in rows] +
        [r'\bottomrule', r'\end{tabularx}', r'\end{table}'])

def figure(name, caption, label, width='0.95'):
    return (r'\begin{figure}[H]\centering'+'\n'+
        r'\includegraphics[width='+width+r'\textwidth]{figures/'+name+'.png}\n'+
        r'\caption{'+caption+'}\n'+r'\label{'+label+'}\n'+r'\end{figure}')

def architecture():
    fig, ax = plt.subplots(figsize=(11, 7.1))
    ax.set(xlim=(0, 11), ylim=(0, 7.1)); ax.axis('off')
    boxes = [
        (.2, 5.55, 5.05, 1.25, 'ERA5-Land regional soil moisture',
         '5 cells; depth-weighted 0-100 cm water\nDaily record: 1981-Apr 2026\nFrozen percentile reference: 1981-2000', '#e8f1f7'),
        (5.75, 5.55, 5.05, 1.25, 'Causal issue-date features',
         'Soil inputs dated at most t - 8 days\nRecent levels, changes, depth and season\nIssue every 5 days in December-March', '#e8f1f7'),
        (.2, 3.65, 5.05, 1.25, 'Chronological model selection',
         'Train: 2001-02 to 2016-17\nValidate: 2017-18 to 2022-23\nEvaluate: 2023-24 to 2025-26', '#e5f1e9'),
        (5.75, 3.65, 5.05, 1.25, 'Selected: histogram gradient boosting',
         '7 / 15 days: max 7 leaves; 30 days: 15\nSoil-only variants selected on validation\nWeather variants compared, not selected', '#e5f1e9'),
        (.2, 1.75, 5.05, 1.25, 'Future regional soil-deficit risk',
         '7-day mean ending at t + 7 / 15 / 30\nDrought <=20th; severe <=10th percentile\nRisk scores are not field diagnoses', '#fff1d6'),
        (5.75, 1.75, 5.05, 1.25, 'Sentinel-2 crop context (parallel)',
         'Frozen sampled points; images <= t - 2\nLatest image <=12 days old, else unknown\nNot an input to the selected soil models', '#f0eafa')]
    for x,y,w,h,title,body,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',fc=color,ec='#8b9aaa'))
        ax.text(x+w/2,y+h-.18,title,ha='center',va='top',fontsize=11,fontweight='bold')
        ax.text(x+w/2,y+.49,body,ha='center',va='center',fontsize=10,linespacing=1.5)
    for a,b in [((5.28,6.17),(5.71,6.17)),((8.28,5.51),(8.28,4.94)),
                ((5.28,4.27),(5.71,4.27)),((8.28,3.61),(2.73,3.04))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.7,'color':'#334e68'})
    ax.text(5.5,.92,'Regional risk + available crop context -> inspection priorities and saved reports',
            ha='center',fontsize=11,fontweight='bold')
    ax.text(5.5,.36,'Independent field / yield validation is still required. Recent evaluation has zero labelled drought cases.',
            ha='center',fontsize=10,color='#8a3333')
    save(fig, 'fig9_rabi_soil_drought_architecture')

def build():
    metrics = pd.read_csv(RES/'evaluation_metrics.csv')
    main = metrics[(metrics.selected == True)&(metrics.group=='pooled')&(metrics.scope=='all')]
    assert len(main)==3 and main.drought_cases.sum()==0 and main.warnings.sum()==0
    back = pd.read_csv(RES/'backtest/metrics.csv')
    selected = back[(back.selected == True)&(back.scope=='all')].sort_values('horizon')
    onset = back[(back.selected == True)&(back.scope=='new_onset')].sort_values('horizon')
    baseline = back[(back.model=='Conditional transition')&(back.scope=='new_onset')].sort_values('horizon')
    cross = pd.read_csv(RES/'cross_product/by_season.csv').query('season >= 2023')
    architecture()

    fig, axes = plt.subplots(1,2,figsize=(10.5,3.5))
    x = np.arange(3)
    for ax,metric,title in zip(axes,['recall','precision'],['New-onset recall','Warning precision']):
        ax.bar(x-.18,100*onset[metric],.36,label='Validation-selected model',color='#305f88')
        ax.bar(x+.18,100*baseline[metric],.36,label='Conditional-transition baseline',color='#cb8427')
        ax.set(xticks=x,xticklabels=['7 days','15 days','30 days'],ylim=(0,100),ylabel='Percent',title=title)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for i,(a,b) in enumerate(zip(onset[metric],baseline[metric])):
            ax.text(i-.18,100*a+2,f'{100*a:.0f}',ha='center',fontsize=9)
            ax.text(i+.18,100*b+2,f'{100*b:.0f}',ha='center',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.04),frameon=False)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'res_r_soil_new_onset')

    fig,ax=plt.subplots(figsize=(8.5,3.2))
    for offset,col,label,color in [(-.18,'era5_drought','ERA5-Land (primary target)','#305f88'),
                                    (.18,'nasa_drought','NASA POWER / MERRA-2 (sensitivity)','#cb8427')]:
        ax.bar(x+offset,cross[col],.36,label=label,color=color)
        for i,v in enumerate(cross[col]):ax.text(i+offset,v+.4,str(v),ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=['2023-24','2024-25','2025-26'],ylim=(0,20),ylabel='Dry cell-dates (out of 125)')
    ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'res_r_soil_product_sensitivity')

    content = r'''\subsection{Regional root-zone soil-moisture drought prediction}
\label{sec:rabi_soil}
The final software adds a distinct drought-related target: \textbf{future regional root-zone soil-water deficit}. The earlier NDVI forecast and early-drying experiment remain separate; their scores cannot be used as the accuracy of this drought model. Government declarations and satellite drying labels are not used as its training target. This is an experimental reanalysis-based risk model, not independently verified crop-loss prediction.

\subsubsection{Data, target and architecture}
Five regional cells supply 82,780 daily ERA5-Land records through April 2026, obtained through Open-Meteo. The approximate 11\,km grid does not resolve individual fields or their irrigation. Root-zone moisture is the depth-weighted mean
\[
\theta_{0:100}=0.07\theta_{0:7}+0.21\theta_{7:28}+0.72\theta_{28:100}.
\]
Seven-day means are converted to empirical percentiles using a frozen 1981--2000 reference and a calendar-day window of $\pm15$ days. A target at or below the 20th percentile is a soil-moisture deficit; at or below the 10th percentile it is severe-or-worse under this definition. Above the 20th percentile means \emph{no percentile-defined deficit}, not a healthy crop.

Issues are every five days from 1 December to 31 March. For horizon $h\in\{7,15,30\}$, the target is the seven-day mean ending at $t+h$; all seven target days are after issue. Soil and weather inputs are dated no later than $t-8$. The reference is fixed before supervised training (2001-02 to 2016-17), validation (2017-18 to 2022-23), and historical evaluation (2023-24 to 2025-26). Dates respect a simulated publication delay, but final reanalysis is not an archive of the original release vintages.

Logistic regression, random forests, histogram gradient boosting (HGB), a soil-analogue model and an equal ensemble are compared with persistence, climatology, conditional transition and damped trend. Validation average precision selects the learned model, with Brier score as a tie-breaker. Warning thresholds maximize validation $F_{0.5}$ subject to at least 50\,\% validation recall; evaluation recall is not constrained. The main selections are soil-only HGB with 7 maximum leaves at 7/15 days and 15 leaves at 30 days. These scores are not asserted to be calibrated probabilities.
'''
    content += figure('fig9_rabi_soil_drought_architecture',
        'Final rabi drought workflow. Satellite imagery supplies parallel crop context; it is not a feature in the selected regional soil-only models. The historical reference, supervised split and availability cutoffs are distinct.', 'fig:r_soil_arch')
    content += r'''\subsubsection{Primary evaluation: no positive drought targets}
\textbf{The 2023-24 to 2025-26 evaluation contains zero drought-labelled targets at every horizon.} The selected models issue zero warnings. Drought recall, average precision and ROC-AUC are therefore not estimable; a no-warning rule also succeeds on these rows. This evaluation cannot demonstrate drought-detection skill, and the reference was not changed to manufacture positive cases.
'''
    content += table('Primary historical evaluation of the soil model. Rows are correlated cell-issue forecasts, not independent drought episodes.',
        'tab:r_soil_primary',['Horizon','Selected model','Rows','Dry targets','Warnings','Recall / AUC'],
        [[int(r.horizon),r.model,int(r.n),int(r.drought_cases),int(r.warnings),'Not estimable'] for r in main.itertuples()], 'lXrrrl')
    content += r'''\subsubsection{Earlier diagnostic backtest: continuation versus new onset}
An additional backtest trains on 2001-02 to 2013-14, validates on 2014-15 to 2016-17, and evaluates 2017-18 to 2019-20. It was chosen after inspecting drought prevalence and is \textbf{post hoc diagnostic evidence}, not an untouched confirmatory test. Its models and thresholds are selected independently on the earlier validation period and differ from the main selections. All positive evaluation targets occur in the 2018-19 season.
'''
    content += table('Earlier diagnostic backtest, all issue dates (including ongoing drought). Precision and recall describe repeated cell-dates.',
        'tab:r_soil_back',['Horizon','Earlier selected model','Caught / dry','Warnings','Precision','Recall'],
        [[int(r.horizon),r.model,f'{round(r.recall*r.drought_cases)} / {int(r.drought_cases)}',int(r.warnings),f'{100*r.precision:.1f}\,\%',f'{100*r.recall:.1f}\,\%'] for r in selected.itertuples()], 'lXrrrr')
    content += r'''The new-onset subset retains only rows whose true issue-day soil indicator is above the 20th percentile. That issue-day truth is used only for retrospective scoring; predictor inputs still stop at $t-8$. There are 337 eligible rows, with only 6, 8 and 15 positive targets at 7, 15 and 30 days. At 15 days, the learned model catches \textbf{3 of 8} positives with 21.4\,\% precision; conditional transition catches \textbf{6 of 8} with 35.3\,\% precision. Thus the selected learned model has not shown an advantage over this simple baseline for new onset.
'''
    content += figure('res_r_soil_new_onset','Post hoc 2017-18 to 2019-20 new-onset comparison. Models were selected on earlier validation data, not these scores. Small, correlated cell-date counts preclude a claim of reliable independent-event performance.','fig:r_soil_onset')
    content += r'''\subsubsection{Dependence on the soil-moisture product}
A separate sensitivity check computes the same reference-period percentile definition using NASA POWER MERRA-2 root-zone wetness (a coarser, rounded product). On the 15-day target dates, ERA5-Land labels no drought cases in the three recent seasons, whereas MERRA-2 labels 15, 2 and 0 of 125 cell-dates respectively. Neither product is field truth. This disagreement limits conclusions about actual crop drought; it is not a reason to select whichever product agrees best with declarations.
'''
    content += figure('res_r_soil_product_sensitivity','Soil-product sensitivity on 15-day target dates. Counts are regional cell-dates, not fields, mandals or independent drought episodes; each season has 125 target rows.','fig:r_soil_products','0.85')
    content += r'''\subsubsection{Operational output and present status}
The runnable workflow saves dated regional 7/15/30-day scores, source dates, input snapshots, provenance hashes, point-level crop context, inspection priorities, GeoJSON and an HTML report. Satellite context requires an image dated at most $t-2$ and no more than 12 days old; missing imagery is \emph{unknown}, not no crop. A 1 March 2026 replay reproduced all 15 saved research predictions and had recent imagery for 1,349 of 1,350 sampled points. The 23 November 2026 run is outside the December--March issue season and correctly produces no seasonal forecasts; current satellite context awaits a fresh export.

The software tests and artifact audit check implementation and reproducibility, not agricultural validity. Independent field-observation templates are supplied but contain no verified field outcomes. \textbf{The software is operational; reliable new agricultural-drought prediction is not yet demonstrated.} The immediate scientific priority is independent water-stress or yield evidence, more drought-bearing seasons/regions, and comparison with conditional transition on the same new-onset population. Future evaluation must preserve missing-data coverage and the distinction between regional water deficit and field crop damage.
'''
    (LATEX/'rabi_drought_section.tex').write_text(content,encoding='utf-8')
    mainpath=LATEX/'main.tex'
    backup=LATEX/'main.pre_drought_v1.tex'
    if not backup.exists():shutil.copy2(mainpath,backup)
    s=mainpath.read_text(encoding='utf-8')
    s=s.replace(r'\section{Module 2b: Rabi Vegetation Forecast and Early-Drying Risk}',r'\section{Module 2b: Rabi Forecasting and Drought-Risk Experiments}')
    if r'\input{rabi_drought_section.tex}' not in s:
        s=s.replace(r'\input{rabi_event_section.tex}',r'\input{rabi_event_section.tex}'+'\n\n'+r'\input{rabi_drought_section.tex}')
    s=s.replace('a separate \\textbf{rabi module} forecasts crop greenness 15 days ahead.',
        'a separate \\textbf{rabi module} studies greenness forecasts, early-drying risk and regional soil-moisture drought prediction.')
    abstract=' The regional rabi soil-moisture model has no drought-positive targets in its recent evaluation; an earlier diagnostic new-onset check does not outperform conditional transition. Reliable agricultural-drought prediction remains unproven.'
    anchor='(historical evaluation).'
    if abstract not in s:s=s.replace(anchor,anchor+abstract,1)
    s=s.replace('\\item \\textbf{Rabi module:} forecast crop greenness (NDVI) 15 days ahead on random cropland points, with rules registered before training.',
        '\\item \\textbf{Rabi module:} distinguish a 15-day NDVI forecast, satellite-defined early-drying risk, and 7/15/30-day regional soil-moisture deficit prediction; evaluate each against its own target and baselines.')
    s=s.replace('This led to the final design: random whole-mandal points and a forecast of greenness rather than a classifier of declarations.',
        'This led to the vegetation-forecast design: random whole-mandal points and a forecast of greenness. The subsequent regional soil-deficit experiment is reported separately in Section~\\ref{sec:rabi_soil}.')
    s=s.replace('Example forecasts at four random points, rabi 2023-24:',
        'Illustrative forecasts at four points randomly sampled within preselected mandal/irrigation groups with sufficient observations, rabi 2023-24:')
    s=s.replace('Feature importance of the final rabi XGBoost model', 'Feature importance of the rabi NDVI-forecast XGBoost model')
    s=s.replace('(temperature, 90-day rain, water balance) together account for only about 11\\,\\% of the gain.',
        '(temperature, 90-day rain, ET$_0$) together account for about 9.5\\,\\% of the gain. Gain importance describes model splits, not a causal effect.')
    s=s.replace('The rabi module therefore forecasts \\textbf{crop greenness}', 'The initial rabi experiment therefore forecasts \\textbf{crop greenness}')
    s=s.replace('Predicted vs observed 15-day NDVI change', 'Predicted vs observed NDVI change from the latest image')
    s=s.replace('predicted and observed 15-day changes:', 'predicted and observed latest-image-to-target changes (15--30 days):')
    s=s.replace('forecasts at four points drawn at random (seed 7) from the',
        'forecasts at four illustrative points drawn with seed 7 within preselected mandal/irrigation groups from the')
    s=s.replace(r'\caption{Training, validation and test periods of the three modules.}',
        r'\caption{Splits for weather, satellite SVI and the rabi NDVI experiment. The separate soil-deficit model uses the longer record and split in Fig.~\ref{fig:r_soil_arch}.}')
    s=s.replace(r'\caption{Operational timing on forecast date $t$: data availability (publication delays) and forecast horizons.}',
        r'\caption{Illustrative availability and target timing. Rabi NDVI horizons start at issue $t$; the regional soil-deficit model is specified separately in Fig.~\ref{fig:r_soil_arch}.}')
    s=s.replace('the selected fields were biased toward resilient fields, which led to random whole-mandal sampling.',
        'the limited candidate-field sample motivated random whole-mandal sampling; resilience of the selected fields was not independently established.')
    s=s.replace('Because rabi (winter) crops depend on stored soil water and irrigation rather than current rain,',
        'Because rabi (winter) water availability can depend on stored soil water, irrigation and in-season rain,')
    s=s.replace('Rabi crops grow on stored soil water and irrigation rather than current rain and often fail late',
        'Rabi crops can depend on stored soil water, irrigation and in-season rain, and may fail late')
    s=s.replace('For rabi, a 15-day NDVI forecast had about 18\\,\\% lower error than the best simple baseline on three held-out seasons.',
        'For rabi, a 15-day NDVI forecast had about 18\\,\\% lower error than the best simple baseline on three historical evaluation seasons. This is vegetation-forecast skill, not drought-detection accuracy.')
    comparison='Always warning on eligible rows catches 107 of 108 events and warns for 453 point-seasons, versus 332 for the model. '
    s=s.replace(comparison, '')
    s=s.replace('These are historical satellite-proxy results, not independently verified drought detection.',
        'Always warning on eligible rows catches 107 of 108 events and warns for 453 point-seasons, versus 332 for the model. These are historical satellite-proxy results, not independently verified drought detection.')
    marker='% RABI-SOIL-CONCLUSION'
    if marker not in s:
        s=s.replace(r'\textbf{Future work:}', marker+'\n'+
            r'The regional soil-moisture extension is reproducible and operational, but the recent evaluation has zero drought-positive targets. In the earlier diagnostic 15-day new-onset subset, the selected model catches 3 of 8 positive cell-dates versus 6 of 8 for conditional transition. Consequently, no reliable agricultural-drought prediction advantage has yet been established.'+'\n\n'+r'\textbf{Future work:}')
    mainpath.write_text(s,encoding='utf-8')

    # Save exact result rows beside the report, and their source hashes.
    evidence=HERE/'rabi_drought_audit';evidence.mkdir(exist_ok=True)
    for name,df in [('main_selected.csv',main),('earlier_selected.csv',selected),('earlier_new_onset.csv',onset),('baseline_new_onset.csv',baseline),('product_comparison.csv',cross)]:df.to_csv(evidence/name,index=False)
    sources=['evaluation_metrics.csv','backtest/metrics.csv','cross_product/by_season.csv','selection_locked.json']
    (evidence/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256((RES/p).read_bytes()).hexdigest() for p in sources},indent=2))
    print('Built soil-model section, three figures and report evidence tables.')

def package():
    with zipfile.ZipFile(HERE/'Final_Report_Overleaf_Updated.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in LATEX.rglob('*'):
            if p.is_file() and '.pre_' not in p.name:z.write(p,p.relative_to(LATEX))
        for p in (HERE/'rabi_drought_audit').glob('*'):z.write(p,'rabi_drought_audit/'+p.name)
        for name in ('REPORT_INFO.md','RABI_FIGURE_AUDIT.md'):
            if (HERE/name).exists():z.write(HERE/name,name)

if __name__=='__main__':
    build();package()
