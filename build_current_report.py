"""Build the current review report from measured CSVs and captured local evidence."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
ORDERED = ROOT / 'results/ordered'
REVIEW = ROOT / 'results/review'
OUT = ROOT / 'PathSnap_Technical_Report.pdf'
INK = colors.HexColor('#14243a')
BLUE = colors.HexColor('#2457d6')
styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=25, leading=30, textColor=INK, spaceAfter=12),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=INK, spaceBefore=14, spaceAfter=7),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=13, textColor=INK, spaceAfter=6),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=7.8, leading=11, textColor=INK, spaceAfter=5),
    'caption': ParagraphStyle('caption', fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=INK, alignment=TA_CENTER, spaceAfter=6),
}


def para(text, style='body'):
    return Paragraph(text, styles[style])


def table(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7.4),
        ('LEADING', (0, 0), (-1, -1), 10), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eef3fa'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#c9d4e5'))
    canvas.line(42, 43, A4[0]-42, 43)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(INK)
    canvas.drawString(42, 30, 'PATHSNAP  |  CURRENT REVIEW DRAFT')
    canvas.drawRightString(A4[0]-42, 30, str(doc.page))


def main():
    old_cv = pd.read_csv(ORDERED / 'cv_summary.csv').set_index('model')
    old_test = pd.read_csv(ORDERED / 'student_test_comparison.csv').set_index('model')
    diagnostics = pd.read_csv(REVIEW / 'five_model_diagnostics.csv').set_index('method')
    eda = pd.read_csv(REVIEW / 'eda_inventory.csv', dtype={'trip_id': str})
    macro = pd.read_csv(REVIEW / 'review_macro_f1.csv').set_index(['split', 'method'])
    timing = pd.read_csv(REVIEW / 'inference_timing.csv').set_index('method')
    continuity = pd.read_csv(REVIEW / 'continuity_live.csv').set_index('method')
    cv = pd.read_csv(REVIEW / 'cv_summary.csv').set_index('method')
    test = pd.read_csv(REVIEW / 'test_summary.csv').set_index('method')
    selection = json.loads((REVIEW / 'selection.json').read_text())
    protocol = json.loads((REVIEW / 'protocol.json').read_text())
    per_trip = pd.read_csv(REVIEW / 'test_per_trip.csv', dtype={'trip_id': str})
    story = []

    story += [para('PathSnap', 'title'),
              para('Technical report draft - measured development results and frozen evaluator integration'),
              para('1. Problem and dataset', 'h1'),
              para('PathSnap assigns noisy GPS observations to directed road arcs, connects successive choices through a directed network, and returns a realistic ordered route. Laptop 1 hosts inference and aggregate scoring through FastAPI; Laptop 2 displays the request and route in a browser.'),
              para('The raw package has 80 training and 10 student-testing trips. Each trip has a three-column GPS .track file, zero-based .nodes, directed zero-based .arcs, and an ordered .route reference when supplied. The 10 evaluator trips are kept separately from training and tuning.'),
              para(f'Raw-file checks found {eda.invalid_gps.sum()} invalid GPS coordinates, {eda.invalid_timestamps.sum()} invalid timestamps and {eda.nonpositive_steps.sum()} nonpositive time steps across 90 development trips. There are {eda.gaps_over_30s.sum()} gaps over 30 seconds. The graphs contain {eda.weak_components.sum()} weak components in total, so the decoder cannot assume every candidate pair connects.'),
              Image(str(REVIEW / 'eda_trajectory_00000052.png'), width=480, height=296),
              para('Student-test trip 00000052: timestamp-coloured GPS observations over its supplied road network.', 'caption'),
              para('2. Preprocessing, targets, and features', 'h1'),
              para('A beam alignment pass assigns each GPS point to an occurrence in the ordered reference route. It uses lateral distance, forward travel and speed plausibility; route rows are never paired one-to-one with GPS rows. All 208,417 training observations retain a proxy label. Candidate generation projects nearby directed arcs within 100 m, keeps up to eight and uses a nearest fallback. A missing reference arc remains a miss in evaluation rather than being injected into candidates.'),
              para('Features include snap distance and position, directed heading, speed and elapsed time, five-point smoothed heading, stationarity, distance relative to the nearest candidate, heading rank, reverse-twin presence, local and whole-trip arc support, node degrees, temporal connectivity, signed offset and 15/30-second context. The reviewed model uses 28 context features. The report files record points far from the reference and a trusted 75 m subset; this is a proxy-label quality check, not a hidden evaluator filter.'),
              para('3. Five learned models', 'h1')]

    methods = ['Conditional logit', 'Grouped MLP', 'ExtraTrees', 'LightGBM ranker', 'HistGradientBoosting']
    rows = [['Learned method', 'CV F1', 'CV top-1', 'Test F1', 'Test top-1', 'Continuity', 'Log loss', 'ms/obs']]
    for name in methods:
        d = diagnostics.loc[name]
        rows.append([name, f'{d.cv_macro_f1:.2%}', f'{old_cv.loc[name,"top1"]:.2%}',
                     f'{d.test_macro_f1:.2%}', f'{old_test.loc[name,"classifier_top1"]:.2%}',
                     f'{d.classifier_continuity:.2%}', f'{old_test.loc[name,"grouped_log_loss"]:.3f}',
                     f'{d.inference_ms_per_observation:.3f}'])
    story += [table(rows, [92, 54, 47, 54, 48, 55, 49, 52]), Spacer(1, 7),
              para('All five methods were compared with two repetitions of eight trip-grouped folds on the 80 training trips. Test means the ten student-test trips. Candidate-row macro F1 marks exactly the highest-scored candidate positive for each observation, then macro-averages positive and negative F1 over all candidate rows. This avoids an arbitrary 0.5 threshold on ranking scores. Continuity is the direct endpoint connection rate between successive classifier-only arc choices. Inference time is model scoring only, measured on the ten student-test trips; feature generation and Viterbi are excluded.', 'small'),
              Image(str(REVIEW / 'five_model_comparison.png'), width=500, height=191),
              para('Measured cross-validation F1, student-test F1, top-1 and grouped loss for the five learned methods.', 'caption')]

    blurbs = [
        ('Conditional logit', 'A linear conditional-choice model minimizes observation-level softmax negative log likelihood with L2 regularisation 1e-4 and L-BFGS-B (45 iterations). It is interpretable and fast, but cannot express nonlinear road interactions. CV top-1 was 76.71%; student-test top-1 was 85.43%. That held-out score is higher than CV, but no training-versus-validation curve was retained, so absence of overfitting cannot be proved.'),
        ('Grouped MLP', 'A 16-unit tanh network minimizes observation-level softmax loss with Adam-style updates for eight epochs. It captures nonlinear interactions but is less transparent and sensitive to scaling. CV top-1 was 77.28%; student-test top-1 was 86.21%. A fresh trip-held-out loss curve is shown below; validation loss flattened while training loss continued downward.'),
        ('ExtraTrees', 'A 24-tree classifier with depth 15 and leaf size 14 uses independent candidate labels and probability scores. It handles nonlinear features but does not train on a grouped objective. CV top-1 was 77.05%; student-test top-1 was 86.29%. Tree depth is constrained, yet the CV/student difference alone does not establish a training-fit gap.'),
        ('LightGBM LambdaRank', 'The initial ranker used 90 trees, 31 leaves, rate 0.08 and LambdaRank on whole observation groups. It had the strongest learned CV top-1 at 78.00% and student-test top-1 at 88.31%. Its strengths are efficient nonlinear ranking and direct candidate-group use; its scores require within-observation calibration for HMM emissions. The later context-feature review is a distinct protocol.'),
        ('HistGradientBoosting', 'The candidate-wise histogram booster used 65 iterations, maximum depth 7, minimum leaf size 50 and L2 0.3. It handles the full data efficiently but its binary objective is not one-choice-per-observation. CV top-1 was 77.77%; student-test top-1 was 87.48%. A 30-trial Bayesian search was recorded separately; pointwise gains did not prove a superior decoded route.'),
    ]
    for name, body in blurbs:
        story.append(KeepTogether([para(name, 'h2'), para(body, 'small')]))
    story += [Image(str(REVIEW / 'grouped_mlp_loss_history.png'), width=420, height=240),
              para('Actual eight-epoch grouped MLP loss on a whole-trip training/validation split.', 'caption')]

    story += [para('4. Bayesian tuning and model choice', 'h1'),
              para('The initial experiment tuned HistGradientBoosting with 30 Optuna TPE trials using trip-held-out grouped validation loss. The later review tuned LightGBM LambdaRank with 18 TPE trials over trees, leaves, depth, rate, minimum leaf samples and L2 regularisation. Trial records and best parameters are retained in results/ordered/optuna_trials.csv and results/review/ranker_optuna_trials.csv. No evaluator trip was used in either search.'),
              Image(str(REVIEW / 'ranker_optuna_history.png'), width=440, height=209),
              para('The 18 LightGBM TPE trial losses and best-so-far objective; the search was limited and its gain was modest.', 'caption'),
              para('The later review reserved 26 training trips for development, including calibration and decoder sweeps, and 52 separate trips for eight-fold assessment. Two training trips with severe reference-distance anomalies were excluded from supervised fitting under the recorded rule. The final learned fit used 180,365 trusted observations with a reference candidate. The deployment now chooses the highest trusted decoded assessment accuracy among trained methods and keeps the rules as explicit baselines.'),
              para(f'The release default is <b>{selection["selected"]} plus Viterbi</b>. It scored {cv.loc["LightGBM untuned", "trusted_decoded_top1"]:.2%} trusted decoded CV accuracy, versus {cv.loc["Nearest road", "trusted_decoded_top1"]:.2%} for nearest road. Their paired interval includes zero, so this is a small and uncertain gain. Optuna tuning improved standalone point accuracy and log loss but reduced decoded CV accuracy: tuned minus untuned was {selection["trusted_cv_tuned_vs_untuned"]["delta"]*100:+.2f} percentage points, with interval [{selection["trusted_cv_tuned_vs_untuned"]["low"]*100:+.2f}, {selection["trusted_cv_tuned_vs_untuned"]["high"]*100:+.2f}]. The choice correction occurred after evaluator data became available; it used the older training CV records only and is not a pristine blind holdout decision.'),
              para('Untuned versus Optuna-tuned LightGBM, same review protocol', 'h2')]
    names = ['LightGBM untuned', 'LightGBM tuned']
    compare = [['Metric', 'Untuned', 'Tuned'],
        ['Trusted CV candidate macro F1', *[f'{macro.loc[("cv_trusted", n),"macro_f1"]:.2%}' for n in names]],
        ['Trusted CV decoded accuracy', *[f'{cv.loc[n,"trusted_decoded_top1"]:.2%}' for n in names]],
        ['Student-test candidate macro F1', *[f'{macro.loc[("student_test", n),"macro_f1"]:.2%}' for n in names]],
        ['Student-test top-1', *[f'{test.loc[n,"point_top1"]:.2%}' for n in names]],
        ['Student-test decoded accuracy', *[f'{test.loc[n,"decoded_top1"]:.2%}' for n in names]],
        ['Student-test covered log loss', *[f'{test.loc[n,"trusted_covered_log_loss"]:.3f}' for n in names]],
        ['Inference ms/observation', *[f'{timing.loc[n,"ms_per_observation"]:.3f}' for n in names]],
    ]
    story += [table(compare, [250, 115, 115]),
              para('Inference timing includes live candidate feature generation and scoring on the same ten student-test trips, but excludes Viterbi. This one-pass measurement is hardware-specific.', 'small'),
              Image(str(REVIEW / 'review_model_comparison.png'), width=500, height=190),
              para('Actual review CV/student F1 and student-test point accuracy; the tuned scorer improves before decoding but loses after decoding.', 'caption'),
              para('5. HMM/Viterbi reconstruction', 'h1'),
              para('A hidden state is a directed candidate arc with a snapped position. The emission is the log of the clipped, normalized within-observation candidate score, scaled by a tuned weight. The transition follows a directed same-arc advance or a shortest path through intermediate arcs, including partial lengths on endpoint arcs. Its log score penalizes |network travel - GPS displacement| / beta, immediate reversal and implausible speed. Viterbi maximizes cumulative emission plus transition score and backtracks one arc per matched observation.'),
              para('Empty candidate sets return unmatched observations. Disconnected pairs create a visible route break. Nonpositive time gaps split segments. A long gap is bridged only when its directed shortest path respects the speed cap. Stationary points favor staying on the same arc. Connecting arcs are returned separately from observation-selected arcs.'),
              para('6. Independent student-test results', 'h1')]

    rows = [['Method', 'Point top-1', 'Decoded top-1', 'Route F1', 'Covered log loss']]
    for name in ['Nearest road', 'Distance + heading', 'LightGBM untuned',
                 'LightGBM tuned', 'HistGradientBoosting']:
        r = test.loc[name]
        rows.append([name, f'{r.point_top1:.2%}', f'{r.decoded_top1:.2%}', f'{r.route_f1:.2%}',
                     f'{r.trusted_covered_log_loss:.3f}'])
    story += [table(rows, [124, 85, 85, 75, 99]), Spacer(1, 6),
              para('These are the ten student-test trips and 19,160 observations, outside training selection. Point accuracy uses ordered proxy labels. Route F1 is twice ordered longest-common-subsequence overlap divided by predicted plus reference route length. Covered log loss excludes points whose reference arc is absent, while top-1 still counts them as errors. The nearest-road baseline rises from 52.95% independent top-1 to 88.51% after decoding; tuning the ranker raises independent top-1 to 91.74% but decodes to 87.87%. Decoder gains vary by trip and are not guaranteed.'),
              para(f'For the deployed untuned LightGBM scorer, independent top-1 is 91.42% and decoded top-1 is 88.53%: Viterbi lowers point accuracy while direct arc continuity rises from {continuity.loc["LightGBM untuned", "classifier_continuity"]:.2%} to {continuity.loc["LightGBM untuned", "decoded_continuity"]:.2%}. On trip 00000007 point accuracy drops from 96.01% to 85.89%; on 00000077 it drops from 82.72% to 70.31%. The emissions and transition constraints are therefore a real trade-off, not a universal accuracy improvement.'),
              PageBreak(), para('7. Folium map and two-laptop demo', 'h1'),
              para('The API lists evaluator trip IDs from an external folder, runs feature generation and frozen inference from .track/.nodes/.arcs, then loads .route only for aggregate server-side metrics. The UI receives GPS points, network and predicted geometry, method labels, metrics, loading errors and route breaks. It never receives reference arc IDs or reference route geometry. Access logs are visible in the Laptop 1 Uvicorn console.')]

    map_image = REVIEW / 'demo_map.png'
    if map_image.is_file():
        story += [Image(str(map_image), width=500, height=333),
              para('Actual development-trip Folium export: GPS, candidate network, classifier-only route, decoded route and reference route are available through labeled layers and a legend.', 'caption')]
    ui_image = REVIEW / 'evaluator_ui_loaded.png'
    if ui_image.is_file():
        story += [PageBreak(), para('8. UI and deployment evidence', 'h1'),
                  Image(str(ui_image), width=500, height=415),
                  para('Local API-connected evaluator UI after a real prediction for trip 00000015. This trip shows a substantial point-accuracy loss after decoding, although continuity rises. A physical two-laptop screenshot and receiving-laptop console photograph must be captured at the event.', 'caption')]
    story += [para('Frozen final smoke tests completed the exposed methods for all ten evaluator IDs over HTTP. Those outcomes were not fed into model selection. The physical same-network two-laptop rehearsal and event screenshots remain to be added; do not present a local screenshot as proof of two separate devices.'),
              para('Evidence: notebooks/PathSnap_Colab.ipynb; MODEL_TRAINING_EVIDENCE.md; results/ordered/; results/review/; pathsnap/review_experiment.py; pathsnap/core.py; pathsnap/api.py; pathsnap/ui.html; EVALUATOR_SETUP.md.', 'small')]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=42, bottomMargin=55, title='PathSnap current technical report draft')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == '__main__':
    main()
