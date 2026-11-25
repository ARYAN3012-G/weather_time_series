"""Update the existing final-report LaTeX from the corrected, saved rabi event audit.

Run after rabi_module/src/detect/causal_events.py. Existing content is backed up
once before the first edit; the result tables are derived, never typed by hand.
"""
from pathlib import Path
import shutil
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RES=ROOT/'rabi_module/results/detect_v2'
LATEX=HERE/'latex'


def escape(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def percent(x):
    return f'{100*x:.1f}' if pd.notna(x) else '--'


def number(x):
    return f'{x:.0f}' if pd.notna(x) else '--'


def table(caption,label,headers,rows,spec):
    return '\n'.join([r'\begin{table}[H]',r'\centering',r'\footnotesize',
        r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabularx}{\textwidth}{@{}'+spec+'@{}}',r'\toprule',
        ' & '.join(r'\textbf{'+h+'}' for h in headers)+r' \\',r'\midrule']+
        [' & '.join(map(str,row))+r' \\' for row in rows]+
        [r'\bottomrule',r'\end{tabularx}',r'\end{table}',''])


def architecture():
    fig,ax=plt.subplots(figsize=(11.6,6.8));fig.patch.set_facecolor('white')
    ax.set_xlim(0,11.6);ax.set_ylim(0,6.8);ax.axis('off')
    colors={'input':'#e8f1f7','model':'#e4f1ec','eval':'#fff1d6'}
    boxes=[(.25,5.1,3.35,1.25,'Sentinel-2 observations','Images dated <= issue - 2 days\nLatest clear image <=12 days old','input'),
           (4.15,5.1,3.35,1.25,'Frozen training reference','2019-20 to 2021-22 only\nOwn season excluded','input'),
           (8.05,5.1,3.3,1.25,'Corrected AgERA5 weather','Weather dated <= issue - 8 days\nAntecedent rain respects cutoff','input'),
           (1.35,3.3,4.2,1.2,'Causal features at issue','Greenness, observed peak and decline\nNormal timing + past weather','model'),
           (6.05,3.3,4.2,1.2,'Models and operating points','LR / RF / XGBoost / ensembles\nWeights + thresholds: validation only','model'),
           (1.35,1.15,4.2,1.35,'Risk of a NEW event','Early drying within 7 / 15 / 30 days\nCandidate warning, not drought diagnosis','eval'),
           (6.05,1.15,4.2,1.35,'Unique-event audit','One point-season = at most one event\nFirst timely warning + coverage + false alerts','eval')]
    for x,y,w,h,title,body,kind in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.1',
                                   linewidth=1.1,edgecolor='#36556c',facecolor=colors[kind]))
        ax.text(x+w/2,y+h-.27,title,ha='center',va='center',fontsize=11,fontweight='bold',color='#173449')
        ax.text(x+w/2,y+.4,body,ha='center',va='center',fontsize=9.5,linespacing=1.5,color='#213745')
    for a,b in [((1.9,5.05),(2.5,4.58)),((5.8,5.05),(4.4,4.58)),((9.7,5.05),(5.1,4.58)),
                ((5.6,3.9),(6.,3.9)),((8.1,3.25),(3.45,2.55)),((5.6,1.8),(6.,1.8))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.4,color='#36556c'))
    ax.text(5.8,.43,'Historical evaluation: 2023-24 to 2025-26 | Already consulted during development',
            ha='center',fontsize=10,color='#465968')
    fig.tight_layout(pad=.4)
    fig.savefig(LATEX/'figures/fig8_rabi_early_drying_architecture.png',dpi=220,bbox_inches='tight')
    fig.savefig(LATEX/'figures/fig8_rabi_early_drying_architecture.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    met=pd.read_csv(RES/'point_date_metrics.csv');events=pd.read_csv(RES/'event_metrics.csv')
    intervals=pd.read_csv(RES/'event_intervals.csv');coverage=pd.read_csv(ROOT/'rabi_module/data/detect_v2/coverage.csv')
    burden=pd.read_csv(RES/'always_warn_comparison.csv')
    rain=json.loads((RES/'rain_association_summary.json').read_text())
    pm=met[(met.season=='pooled')&met.selected].sort_values('horizon')
    em=events[(events.season=='pooled')&events.selected].sort_values('horizon')
    chosen30=em[em.horizon==30].iloc[0]
    architecture()
    parts=[r'\subsection{Early-drying risk and a unique-event audit}',
        r'\label{sec:rabi_events}',
        r'This additional experiment predicts \textbf{new satellite-defined early drying}, not government declarations '
        r'or independently verified agricultural drought. An event is a post-peak greenness crossing at least 21 days '
        r'earlier than the point\textquotesingle s historical median drying date. Crop identity and actual harvest '
        r'dates are not verified. Early drying is therefore a candidate stress signal; it can also reflect crop or '
        r'management changes. The 2023--2026 seasons were already consulted during development.',
        r'\subsubsection{Corrections before the reported run}',
        r'The first classifier supplied a completed-season early-drying status as an input, used the validation '
        r'season in some reference quantities, and included unpublished late-November rainfall at the earliest '
        r'December issues. Its high status-classification accuracy and repeated-row warning percentages are '
        r'\textbf{superseded}. Corrected version v2 removes the retrospective status input, builds references only '
        r'from training seasons, and enforces the eight-day weather cutoff for every weather feature. '
        r'Its target is an event in $(t,t+h]$, not a status that remains positive after drying.',
        r'\subsubsection{Causal inputs and model comparison}',
        r'Training uses 2019-20 to 2021-22, validation uses 2022-23, and historical evaluation uses 2023-24 to '
        r'2025-26. Each training season is excluded from its own reference. At least two established training '
        r'seasons with observed drying dates are required. The stricter reference changes the eligible population; '
        r'v1 and v2 are not a same-cohort ablation. Forecasts use only observations dated at most $t-2$ days, with '
        r'the latest at most 12 days old, and weather dated at most $t-8$ days. A drying flag observed by issue '
        r'time is computed from the observation prefix; it is distinct from the retrospective outcome.',
        r'Logistic regression, random forest and XGBoost are compared with equal and validation-weighted '
        r'probability ensembles. Random forest and XGBoost use five seeds; logistic regression is deterministic. '
        r'The primary model is selected by validation average precision, and thresholds maximize validation '
        r'balanced accuracy. Baseline imputations, weights and thresholds never use evaluation data. '
        r'Tables report the seed-averaged deployment predictions; the reproducibility files retain seed metrics.',
        r'\begin{figure}[H]',r'\centering',
        r'\includegraphics[width=0.98\textwidth]{figures/fig8_rabi_early_drying_architecture.png}',
        r'\caption{Corrected rabi early-drying risk workflow. Retrospective labels are used for scoring, not as inputs.}',
        r'\label{fig:rabi_event_arch}',r'\end{figure}',
        table('Reference and observation coverage for the corrected experiment. Points are sampled locations, not verified field boundaries.',
              'tab:r_event_coverage',['Season','Reference eligible','Scorable','Forecast points','Early events'],
              [[f'{int(r.season)}-{str(int(r.season)+1)[2:]}',number(r.reference_ok),number(r.scorable),
                number(r.forecast_points),number(r.early_events)] for r in coverage.itertuples()], 'Xrrrr'),
        r'\subsubsection{New-onset point-date performance}',
        r'Only forecasts preceding the retrospective event and without an issue-time observed drying flag '
        r'enter this comparison. These are correlated point-dates, not independent events. Average precision '
        r'(AP) is reported alongside ROC-AUC because new onsets are uncommon.',
        table('New-onset point-date scores for models selected on validation. Precision and recall are percentages.',
              'tab:r_onset_metrics',['Days','Selected model','Rows (positive)','AUC','AP','Precision','Recall'],
              [[str(r.horizon),escape(r.model),f'{r.n} ({r.positives})',f'{r.roc_auc:.3f}',f'{r.average_precision:.3f}',
                percent(r.precision),percent(r.recall)] for r in pm.itertuples()], 'rXrrrrr'),
        r'\subsubsection{One count per drying event}',
        r'An event is one sampled point-season with an observed early-drying date. For horizon $h$, a timely '
        r'warning must precede the event by more than zero and no more than $h$ days. Repeated warnings cannot '
        r'increase the count of caught events. The denominator includes all monitored events, including events '
        r'without a usable advance observation. Opportunity recall is additionally reported in the source tables.',
        r'Warning precision in Table~\ref{tab:r_unique_events} is the fraction of \emph{warned point-seasons} '
        r'with at least one timely warning. A point-season with earlier false alerts and a later timely alert '
        r'counts once as successful, so this is not per-alert precision. Table~\ref{tab:r_onset_metrics} retains '
        r'the point-date precision. Lead time uses the first timely warning for each caught event, once.',
        table('Unique early-drying events. Recall uses all monitored events; precision uses warned point-seasons. Lead is the median first timely warning.',
              'tab:r_unique_events',['Days','Selected model','Caught / events','With opportunity','Recall (\%)','Precision (\%)','Lead (d)'],
              [[str(r.horizon),escape(r.model),f'{r.caught_events} / {r.monitored_events}',str(r.events_with_opportunity),
                percent(r.recall_all_events),percent(r.warning_precision),number(r.median_lead)] for r in em.itertuples()], 'rXrrrrr'),
        table('Recall requiring a minimum useful lead, among all monitored early-drying events (percent).',
              'tab:r_useful_lead',['Days','At least 7 d lead','At least 14 d lead','Lead IQR (d)'],
              [[str(r.horizon),percent(r.recall_at_least_7_days),percent(r.recall_at_least_14_days),
                f'{number(r.lead_q25)}--{number(r.lead_q75)}'] for r in em.itertuples()], 'Xrrr'),
        r'\subsubsection{Uncertainty and interpretation}',
        table('95\% bootstrap intervals for unique-event recall and warned-point-season precision. Cell intervals are indicative with very few clusters.',
              'tab:r_event_ci',['Days','Resampling unit','Clusters','Recall interval (\%)','Precision interval (\%)'],
              [[str(r.horizon),escape(r.bootstrap_unit),str(r.clusters),
                f'{percent(r.recall_all_events_lo)}--{percent(r.recall_all_events_hi)}',
                f'{percent(r.warning_precision_lo)}--{percent(r.warning_precision_hi)}'] for r in intervals.itertuples()], 'rXrrr'),
        r'The event ledger and full model/baseline tables are saved under '
        r'\texttt{rabi\_module/results/detect\_v2/}; operating points and validation predictions are retained. '
        r'Rainfall association is a descriptive cell-season consistency check, not independent field validation. '
        r'The reference spans only two or three seasons, and cloud gaps and retrospective smoothing make the '
        r'crossing date an image-based estimate rather than a measured physiological onset. No observed crossing '
        r'does not prove a crop was free of water stress. Before claiming drought detection, alerts require '
        r'independent evidence on crop water stress and crop-management timing.','']
    parts.extend([r'\subsubsection{Rainfall association}',
        f'The corrected reference yields {rain["cell_seasons"]} cell-seasons with at least ten established points. '
        f'The Spearman correlation of early-drying share with June--November rain relative to normal is '
        f'{rain["spearman"]:.3f}. This is descriptive historical association, not causal or field-level validation. '
        f'There are {rain["dry_cell_seasons"]} qualifying cell-seasons below 80\\,\\% of normal rain, so the '
        r'low-rain versus other-years rate contrast cannot be estimated under that rule. The previous '
        r'``twice as common'' statement is not carried forward to this corrected cohort.',
        r'\subsubsection{Always-warn sanity check}',
        r'This diagnostic was added after viewing the primary v2 results and did not change any model, threshold '
        r'or selection. Warning at every usable issue also catches almost all events. Thus event recall and '
        r'first-timely lead cannot be interpreted without the warning burden. The learned model reduces alerts, '
        r'but many warned point-seasons still have no timely event.',
        table('Selected model versus warning at every usable issue. One warned unit is one point-season. '
              'This diagnostic is not used to choose the model.', 'tab:r_always_warn',
              ['Days','Model caught','Always caught','Model warned','Always warned'],
              [[str(r.horizon),f'{r.model_caught}/{r.total_events}',f'{r.always_caught}/{r.total_events}',
                str(r.model_warned_units),str(r.always_warned_units)] for r in burden.itertuples()], 'Xrrrr')])
    section='\n\n'.join(parts)
    (LATEX/'rabi_event_section.tex').write_text(section,encoding='utf-8')
    main=LATEX/'main.tex';backup=LATEX/'main.pre_detect_v2.tex'
    if not backup.exists():shutil.copy2(main,backup)
    text=main.read_text(encoding='utf-8')
    anchor=r'\section{Module 3: UAV Drone Drought Segmentation}'
    if r'\input{rabi_event_section.tex}' not in text:
        text=text.replace(anchor,r'\input{rabi_event_section.tex}'+'\n\n'+anchor)
    text=text.replace(r'\section{Module 2b: Rabi Vegetation Forecast}',r'\section{Module 2b: Rabi Vegetation Forecast and Early-Drying Risk}')
    wide=(r'\begin{tabular}{@{}lcccccc@{}}'+'\n'+r'\toprule'+'\n'+
          r'\textbf{Population} & \textbf{XGBoost (main)}')
    if r'\setlength{\tabcolsep}{4pt}'+'\n'+wide not in text:
        text=text.replace(wide,r'\setlength{\tabcolsep}{4pt}'+'\n'+wide)
    text=text.replace('All modules are evaluated on held-out years, with calibrations frozen on training periods, purged chronological splits, automated leakage tests, strong simple baselines and rules fixed before evaluation; government drought orders are used only as an independent comparison.',
        'The weather and satellite studies use chronological evaluation and documented calibration and data-availability controls. Rabi experiments are historical development evaluations on seasons already examined, with documented corrections and simple baselines; government drought orders are an external comparison, not rabi training labels.')
    text=text.replace(r'All rules were written and committed to git \emph{before} any model was trained; two independent reviews led to documented amendments',
        r'The initial forecasting protocol was committed before the first training run; subsequent reviews led to documented amendments')
    text=text.replace(r'\item Rabi: evaluated on seasons known during design; it forecasts NDVI, not drought; points are sampled pixels, not verified fields; only five weather cells and three test seasons.',
        r'\item Rabi: the NDVI forecast and early-drying classifier use seasons already examined during development. Early-drying labels are satellite proxies, not confirmed drought. Points are sampled pixels, not verified fields; crop type and harvest dates are unverified; very few weather cells and only three evaluation seasons limit inference.')
    text=text.replace(r'\item \textbf{Weather adds little to crop forecasts.} In both kharif (15--30 days) and rabi (15 days), crop condition is set mostly by water already lost; satellite history carries most of the forecast skill.',
        r'\item \textbf{Weather contribution depends on the task.} The tested weather inputs did not improve the pooled 15-day rabi NDVI forecast. This ablation does not establish that water availability is unimportant or identify the physical cause of crop stress.')
    # Add an accurate, generated conclusion sentence without replacing other modules' results.
    marker='% RABI-EVENT-CONCLUSION'
    if marker in text:
        start=text.index(marker)
        end=text.index(r'\textbf{Future work:}',start)
        text=text[:start]+text[end:]
    sentence=(f' In the corrected early-drying experiment, the validation-selected 30-day model warned for '
            f'{int(chosen30.caught_events)} of {int(chosen30.monitored_events)} monitored point-season events '
            f'({percent(chosen30.recall_all_events)}\\,\\%); warned-point-season precision was '
            f'{percent(chosen30.warning_precision)}\\,\\%, with median first timely lead '
            f'{number(chosen30.median_lead)} days. These are historical satellite-proxy results, not independently verified drought detection.')
    target=r'\textbf{Future work:}'
    text=text.replace(target,marker+'\n'+sentence+'\n\n'+target)
    main.write_text(text,encoding='utf-8')
    info=HERE/'REPORT_INFO.md';iback=HERE/'REPORT_INFO.pre_detect_v2.md'
    if not iback.exists():shutil.copy2(info,iback)
    s=info.read_text(encoding='utf-8');head='\n## Corrected early-drying event audit (22 November 2026)\n'
    if head in s:s=s[:s.index(head)]
    excerpt=(RES/'EVENT_REPORT.md').read_text(encoding='utf-8').split('## All models and baselines:')[0]
    info.write_text(s+head+'\nSource: `rabi_module/results/detect_v2/EVENT_REPORT.md`. '
        'Supersedes classifier v1 scores; keep the separate frozen NDVI forecasting result.\n\n'+excerpt,encoding='utf-8')
    print('Updated final-report LaTeX, generated tables, architecture figure and facts sheet.')


if __name__=='__main__':main()
