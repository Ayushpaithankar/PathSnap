"""Build the historical first-pass PDF; use build_current_report.py for the current report."""
import csv
import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from pathsnap.core import load_reference, load_trip


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
INK = colors.HexColor('#182b3c')
ACCENT = colors.HexColor('#08736d')
STYLES = {
    'title': ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=30, leading=35, textColor=INK),
    'section': ParagraphStyle('Section', fontName='Helvetica-Bold', fontSize=20, leading=25, textColor=INK),
    'label': ParagraphStyle('Label', fontName='Helvetica-Bold', fontSize=9, leading=14, textColor=ACCENT),
    'body': ParagraphStyle('Body', alignment=TA_LEFT, fontName='Helvetica', fontSize=10, leading=15,
                           textColor=INK, spaceAfter=9),
}


def records(name):
    with (RESULTS / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def table(rows, widths):
    value = Table(rows, colWidths=widths, hAlign='LEFT')
    value.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eef5f6'), colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7), ('LEFTPADDING', (0, 0), (-1, -1), 9),
    ]))
    return value


def trip_figure():
    dataset = ROOT / 'inputs/Student folder.zip'
    track, nodes, arcs = load_trip(dataset, 'student_test', '00000052')
    reference = load_reference(dataset, 'student_test', '00000052')
    checkpoint = json.loads((RESULTS / 'checkpoint_00000052.json').read_text())
    chosen = [r['decoded_arc_id'] for r in checkpoint['predictions'] if r['decoded_arc_id'] is not None]
    connectors = [a for r in checkpoint['predictions'] for a in r['connecting_arc_ids']]
    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.3), layout='constrained')
    def geometry(ids):
        return [[nodes[i] for i in arcs[a]] for a in set(ids)]
    for ax, path, color, title in zip(
        axes, [reference, chosen], ['#c36520', '#08736d'], ['Reference route', 'Decoded route']
    ):
        ax.add_collection(LineCollection(geometry(range(len(arcs))), colors='#ccd3d9', linewidths=.35))
        ax.add_collection(LineCollection(geometry(path), colors=color, linewidths=1.5))
        if ax is axes[1]:
            ax.add_collection(LineCollection(geometry(connectors), colors='#573aba', linewidths=1.3,
                                             linestyles='dashed'))
        lon = [o['lon'] for o in track]
        lat = [o['lat'] for o in track]
        ax.scatter(lon, lat, s=3, c='#2466a9', alpha=.8)
        ax.set(xlim=(min(lon)-.001, max(lon)+.001), ylim=(min(lat)-.001, max(lat)+.001),
               title=title, xlabel='Longitude')
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Latitude')
    output = BytesIO()
    figure.savefig(output, format='png', dpi=190)
    plt.close(figure)
    output.seek(0)
    return Image(output, width=500, height=204)


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#b9c9ce'))
    canvas.line(46, 45, A4[0]-46, 45)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(INK)
    canvas.drawString(46, 31, 'PATHSNAP  /  TECHNICAL REPORT')
    canvas.drawRightString(A4[0]-46, 31, str(doc.page))


def main():
    frozen = json.loads((RESULTS / 'decoder_summary.json').read_text())
    live = json.loads((RESULTS / 'decoder_summary_live.json').read_text())
    checkpoints = records('decoder_per_trip.csv')
    comparison = records('model_comparison.csv')
    inventory = records('data_quality_summary.csv')
    pct = lambda value: f'{float(value):.2%}'
    story = []

    def p(text, name='body'):
        story.append(Paragraph(text, STYLES[name]))

    p('TECHNICAL REPORT  /  GPS MAP MATCHING', 'label')
    p('PathSnap', 'title')
    story.append(Spacer(1, 12))
    p('PathSnap scores directed road arcs for noisy GPS observations, reconstructs a feasible route and serves the result to a browser through FastAPI.')
    p('Dataset and feature design', 'section')
    p('The development package contains 80 training trips and 10 student test trips. Each trip supplies GPS timestamps, node coordinates and directed arcs. The route file provides an ordered reference for evaluation. Trip IDs and directed arc row IDs are preserved from the source files.')
    story.append(table([['Partition', 'Observations', 'Label coverage', 'Candidate miss rate']] +
        [[r['partition'], r['observations'], pct(r['label_coverage']), pct(r['candidate_miss_rate'])]
         for r in inventory], [138, 120, 116, 126]))
    story.append(Spacer(1, 12))
    p('A local metric projection supports a spatial search for up to five arcs within 75 m, with a nearest-arc fallback. The classifier uses distance, arc length and position, heading agreement, GPS speed and time gap, candidate rank and candidate count.')
    p('Classifier selection', 'section')
    p('The supplied run sampled up to 250 training observations per trip and used four grouped folds that held out whole trips. Model selection used mean cross-validation macro F1. The search for HistGradientBoosting used eight Bayesian trials and selected a learning rate of 0.07247 and 29 maximum leaves.')
    story.append(table([['Classifier', 'CV macro F1', 'Student top-1', 'Student log loss']] +
        [[r['model'] + (' (tuned)' if r['phase'] == 'tuned' else ''), f"{float(r['cv_macro_f1']):.4f}",
          pct(r['student_top1']), f"{float(r['student_log_loss']):.4f}"] for r in comparison],
        [203, 93, 100, 104]))
    story.append(Spacer(1, 11))
    p('The release retains the supplied tuned model and its matching score export. A separate local rerun completed preparation, model comparison and tuning, and selected the untuned configuration by CV. Both runs are recorded in results/.')

    story.append(PageBreak())
    p('ROUTE RECONSTRUCTION', 'label')
    p('Directed Viterbi decoder', 'section')
    p('Each state represents a directed candidate arc and its snapped position. Emissions are the logs of clipped positive-class scores after normalization within each observation. The transition distance combines the remaining length of the previous arc, a directed shortest connector and the prefix of the next arc. The transition penalty is |network distance - GPS displacement| / 100. The travel cap is 45 m/s times the time gap plus 100 m.')
    p('On the same arc, backward drift within 8 m is treated as stationary GPS jitter. Larger backward movement requires a directed return path. Dijkstra caches successful shortest paths and checks the exact bound on every reuse. Viterbi starts a new segment across a gap over 30 s, a nonpositive time gap or a disconnected transition. Empty candidate sets produce unmatched observations.')
    p('Student test results', 'section')
    p(f"The scoring check matched all {frozen['labeled']:,} geometric labels out of {frozen['observations']:,} observations across the ten trips. Reference alignment projects GPS points onto reference arcs and adds a heading penalty when motion supports it. It rejects projections beyond 45 m or within 1 m of a competing cost. Candidate misses remain in the pointwise denominator.")
    rows = [['Metric', 'Classifier only', 'Viterbi (frozen)', 'Viterbi (live)']]
    for key, label in [('point_accuracy','Point accuracy'), ('direct_continuity','Direct-neighbor continuity'),
                       ('sequence_f1','Ordered route overlap F1')]:
        rows.append([label, pct(frozen['baseline'][key]), pct(frozen['decoded'][key]), pct(live['decoded'][key])])
    story.append(table(rows, [186, 106, 104, 104]))
    story.append(Spacer(1, 10))
    p('Direct continuity counts adjacent observation arcs that match or share a directed endpoint. Ordered overlap F1 is twice the longest common subsequence length divided by the sum of the sequence lengths. Predicted runs collapse consecutive repeat IDs and include the decoder connectors. The reference stays in its original order.')
    story.append(trip_figure())
    p('Trip 00000052. Blue points show GPS, grey lines show the network and purple dashed lines show directed connectors. The bundled Folium map provides independently toggleable layers.')

    story.append(PageBreak())
    p('VALIDATION AND DEPLOYMENT', 'label')
    p('Per-trip results and demo', 'section')
    rows = [['Student test trip', 'Eligible points', 'Classifier accuracy', 'Viterbi accuracy', 'Viterbi route F1']]
    rows += [[r['trip_id'], r['labeled'], pct(r['baseline_point_accuracy']),
              pct(r['decoded_point_accuracy']), pct(r['decoded_sequence_f1'])] for r in checkpoints]
    story.append(table(rows, [97, 83, 110, 105, 105]))
    story.append(Spacer(1, 12))
    p('Trip 00000052 gained about 4.22 percentage points in point accuracy. The decoder lost about 1.60 points on 00000007 and 1.68 points on 00000056. The alignment is a geometric proxy, and plausible competing roads remain difficult at intersections and overlaps.')
    p(f"Frozen candidate coverage differs from live coverage at {live['candidate_coverage_differences']} observation(s). "
      f"Live point accuracy is {pct(live['baseline']['point_accuracy'])} for the classifier and {pct(live['decoded']['point_accuracy'])} for Viterbi. The frozen candidate miss rate is {pct(frozen['candidate_miss_rate'])}.")
    p('Demo and verification', 'section')
    p('Laptop 1 runs FastAPI with access logs. Laptop 2 opens the service URL on the same LAN. The server reads the bundled ZIP or an extracted dataset, computes candidates live and returns GPS points, predicted road geometry, connectors, and aggregate metrics. Reference routes are loaded after inference and stay on the server.')
    p('Local API smoke tests covered all ten student test trips and a missing-trip response. The browser successfully rendered a live prediction. The physical LAN rehearsal and reserved evaluator runs remain pending the required devices and data.')
    p('Evidence and provenance', 'section')
    p('The archive includes the fitted model, fold records and tuning trials, frozen and live decoder score tables, the live HTTP check, and the interactive development map. Member 1 supplied the classifier and original experiment evidence. Codex assisted with integration, verification and report writing.')
    doc = SimpleDocTemplate(str(ROOT / 'PathSnap_Technical_Report_Legacy.pdf'), pagesize=A4,
                            leftMargin=46, rightMargin=46, topMargin=46, bottomMargin=57,
                            title='PathSnap Technical Report')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    main()
