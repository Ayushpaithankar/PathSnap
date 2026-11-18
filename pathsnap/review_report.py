"""Summarize the independent review experiment from its saved evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .review_experiment import HGB, NO_CONTEXT, RANKER, RULES, UNTUNED


def main():
    root = Path('results/review')
    protocol = json.loads((root / 'protocol.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    decoder = json.loads((root / 'decoder_parameters.json').read_text())['configs']
    cv = pd.read_csv(root / 'cv_summary.csv').set_index('method')
    test = pd.read_csv(root / 'test_summary.csv').set_index('method')
    per_trip = pd.read_csv(root / 'test_per_trip.csv', dtype={'trip_id': str})
    intervals = pd.read_csv(root / 'trip_bootstrap.csv')
    ablation = pd.read_csv(root / 'emission_ablation.csv')
    audit = pd.read_csv(root / 'ranker_error_audit.csv', dtype={'trip_id': str})
    deployed_audit = pd.read_csv(root / 'deployed_error_audit.csv', dtype={'trip_id': str})
    profiles = pd.read_csv(root / 'decoder_final_profiles.csv')
    out = ['# PathSnap review validation', '',
           f'**Release default: {selection["selected"]}.** Both the release default and the tuned ranker '
           'are available in the dashboard when they differ.', '',
           f'The reference-distance rule excluded {", ".join(protocol["excluded_trips"])} from training '
           f'({protocol["excluded_observations"]:,} observations). The reviewed split contains '
           f'{len(protocol["development_trips"])} development trips and '
           f'{len(protocol["assessment_cv_trips"])} separate eight-fold assessment trips. '
           f'The final supervised model fit used {selection["trusted_fit_observations"]:,} trusted points with a reference candidate.', '',
           '| Method | Trusted CV point top-1 | Trusted CV decoded | Covered CV log loss | Test point top-1 | Test decoded | Test route F1 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, row in test.iterrows():
        k = cv.loc[name]
        out.append(f'| {name} | {k.trusted_point_top1:.2%} | {k.trusted_decoded_top1:.2%} | {k.trusted_covered_log_loss:.3f} | '
                   f'{row.point_top1:.2%} | {row.decoded_top1:.2%} | {row.route_f1:.2%} |')
    coverage = cv.iloc[0]
    out += ['', f'Assessment coverage: {int(coverage.trusted_observations):,} trusted observations out of '
            f'{int(coverage.observations):,}, with {int(coverage.trusted_misses):,} trusted candidate misses. '
            'Covered log loss excludes misses; accuracy still counts them as errors. The all-point loss with '
            'the 1e-12 floor remains in the CSV for comparison.', '',
            '## Paired uncertainty', '',
            '| Split | Tuned ranker minus | Decoded difference | 95% trip interval | Wins / losses / ties |',
            '| --- | --- | ---: | ---: | ---: |']
    for r in intervals.itertuples():
        out.append(f'| {r.split} | {r.baseline} | {r.delta*100:+.2f} pp | [{r.low*100:+.2f}, {r.high*100:+.2f}] pp '
                   f'| {r.wins} / {r.losses} / {r.ties} |')
    rng = np.random.default_rng(2027)
    ranker = per_trip[per_trip.method == RANKER].sort_values('trip_id')
    indexed = rng.integers(0, len(ranker), size=(10000, len(ranker)))
    route = lambda frame: (2 * frame.route_matches.to_numpy()[indexed].sum(axis=1) /
                           frame.sequence_total.to_numpy()[indexed].sum(axis=1))
    nearest = per_trip[per_trip.method == RULES[0]].sort_values('trip_id')
    lo, hi = np.quantile(route(ranker) - route(nearest), [.025, .975])
    gain = test.loc[RANKER, 'route_f1'] - test.loc[RULES[0], 'route_f1']
    out += ['', f'The test route F1 difference (tuned ranker minus nearest) is {gain*100:+.2f} pp, '
            f'with a paired trip interval of [{lo*100:+.2f}, {hi*100:+.2f}] pp. '
            'Point accuracy uses aligned proxy labels. Route overlap uses the original ordered arc list.', '',
            '## What tuning and context contributed', '']
    for name in (UNTUNED, NO_CONTEXT, HGB):
        out.append(f'- Compared with **{name}**, the tuned ranker changes test point accuracy by '
                   f'{(test.loc[RANKER,"point_top1"]-test.loc[name,"point_top1"])*100:+.2f} pp and '
                   f'decoded accuracy by {(test.loc[RANKER,"decoded_top1"]-test.loc[name,"decoded_top1"])*100:+.2f} pp.')
    out += ['', '### Emission ablation under the tuned ranker decoder', '',
            '| Emissions | Test decoded top-1 | Route F1 |', '| --- | ---: | ---: |']
    for row in ablation.itertuples():
        out.append(f'| {row.method} | {row.decoded_top1:.2%} | {row.route_f1:.2%} |')
    out += ['', '## Decoder search', '',
            '| Method | Emission weight | Beta multiplier | Gap threshold (s) | Speed cap (m/s) |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name, d in decoder.items():
        out.append(f'| {name} | {d["emission_weight"]} | {d["beta_multiplier"]} | {d["max_gap_s"]} | {d["max_speed_mps"]} |')
    out += ['', 'The local profiles below hold the other coordinates fixed. An edge or flat profile does not establish an interior optimum.', '',
            '| Method | Coordinate | Selected | Best probed value(s) | Selected accuracy | Best accuracy |',
            '| --- | --- | ---: | --- | ---: | ---: |']
    for (name, axis), group in profiles.groupby(['method', 'coordinate'], sort=False):
        selected = group[group.selected].iloc[0]
        best = group.accuracy.max()
        plateau = ', '.join(f'{v:g}' for v in group[group.accuracy == best].value)
        out.append(f'| {name} | {axis} | {selected.value:g} | {plateau} | {selected.accuracy:.2%} | {best:.2%} |')
    out += ['', '## Tuned ranker decoded error audit', '',
            '| Exclusive category | Count | Share |', '| --- | ---: | ---: |']
    for key, count in audit.category.value_counts().items():
        out.append(f'| {key} | {count:,} | {count/len(audit):.1%} |')
    for t in ('00000077', '00000007'):
        errors = audit[audit.trip_id == t]
        row = per_trip[(per_trip.method == RANKER) & (per_trip.trip_id == t)].iloc[0]
        reverse = errors[errors.category == 'reverse arc']
        out += ['', f'Trip {t} has {len(errors)} decoded errors. Of {len(reverse)} reverse errors, '
                f'{int(reverse.reverse_predicted_agrees_with_gps.sum())} have local GPS displacement along the prediction. '
                f'Point top-1 is {row.point_correct/row.observations:.2%}; decoded top-1 is {row.decoded_correct/row.observations:.2%}.']
    lost = audit[(audit.trip_id == '00000007') & audit.ranker_was_correct].observation_index.to_numpy()
    runs = np.split(lost, np.where(np.diff(lost) != 1)[0] + 1)
    longest = max(runs, key=len)
    if len(longest):
        out += ['', f'The longest uninterrupted loss run on trip 7 spans indices {longest[0]} through {longest[-1]} '
                f'({len(longest)} observations). Direction agreement is a diagnostic, since heading noise and reference errors can both contribute.']
    out += ['', f'## Release-default decoded error audit ({selection["selected"]})', '',
            '| Exclusive category | Count | Share |', '| --- | ---: | ---: |']
    for key, count in deployed_audit.category.value_counts().items():
        out.append(f'| {key} | {count:,} | {count/len(deployed_audit):.1%} |')
    (root / 'RESULTS.md').write_text('\n'.join(out) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
