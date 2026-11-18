"""Build a concise Markdown report from the saved ordered-label experiments."""
import ast
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path('results/ordered')
    model_config = json.loads((root / 'configuration.json').read_text())
    current = model_config['model']
    train = pd.read_csv(root / 'cv_summary.csv').set_index('model')
    decoder = pd.read_csv(root / 'decoder_cv_summary.csv').set_index('model')
    test = pd.read_csv(root / 'student_test_comparison.csv').set_index('model')
    per_trip = pd.read_csv(root / 'student_test_per_trip.csv', dtype={'trip_id': str})
    tuning = pd.read_csv(root / 'optuna_trials.csv')
    audit = pd.read_csv(root / 'alignment_audit.csv', dtype={'trip_id': str})
    original = json.loads(Path('results/legacy_error_audit/summary.json').read_text())['methods']
    c = model_config
    comparison = [
        '| Method | CV point top-1 | CV grouped loss | CV decoded top-1 | Test point top-1 | Test grouped loss | Test decoded top-1 | Test route F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, row in test.iterrows():
        label = f'**{name}**' if name == current else name
        cv = train.loc[name]
        comparison.append(f'| {label} | {cv.top1:.2%} | {cv.grouped_log_loss:.3f} | '
                          f'{decoder.loc[name].decoded_accuracy:.2%} | {row.classifier_top1:.2%} | '
                          f'{row.grouped_log_loss:.3f} | {row.decoded_top1:.2%} | {row.sequence_f1:.2%} |')
    evidence = [
        '| Trip | Labeled points | Candidate misses | Nearest road | Distance + heading | Model top-1 | Decoded top-1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for trip_id, chunk in per_trip.groupby('trip_id'):
        rows = chunk.set_index('model')
        row = rows.loc[current]
        evidence.append(f'| {trip_id} | {int(row.observations):,} | {int(row.candidate_misses)} | '
                        f'{rows.loc["Nearest road"].classifier_top1:.2%} | '
                        f'{rows.loc["Distance + heading"].classifier_top1:.2%} | '
                        f'{row.classifier_top1:.2%} | {row.decoded_top1:.2%} |')
    d = c['decoder']
    major = audit[audit.partition == 'train'].nlargest(2, 'candidate_misses')
    zero = test.loc['Nearest road']
    selected = test.loc[current]
    cv_best = c['best_overall_cv_method']
    grouped_gain = train.loc[current, 'paired_top1_delta_vs_rule']
    grouped_spread = train.loc[current, 'paired_top1_delta_sd']
    early = tuning[tuning.state == 'COMPLETE']
    stops = sum(any(n < row.params_max_iter for n in ast.literal_eval(row.user_attrs_iterations_grown))
                for row in early.itertuples())
    legacy = pd.read_csv(root / 'legacy_on_ordered_labels.csv').sum(numeric_only=True)
    trip_77 = per_trip[(per_trip.trip_id == '00000077') & (per_trip.model == current)].iloc[0]
    selected_trips = per_trip[per_trip.model == current].copy()
    selected_trips['decoder_delta'] = selected_trips.decoded_top1 - selected_trips.classifier_top1
    largest_drop = selected_trips.loc[selected_trips.decoder_delta.idxmin()]
    output = [
        '# PathSnap ordered-label experiments',
        '',
        f'The deployed learned model is **{current}**. Decoder CV selected it among models trained on a grouped objective. '
        f'The best overall training-fold combination was **{cv_best} + Viterbi** '
        f'({decoder.loc[cv_best].decoded_accuracy:.2%} versus {decoder.loc[current].decoded_accuracy:.2%} for the deployed model). '
        f'On the student test split, the deployed model decodes to {selected.decoded_top1:.2%}; '
        f'nearest-road scores plus the same decoder reach {zero.decoded_top1:.2%}. A small holdout gap should be interpreted with the CV ranking.',
        '',
        *comparison,
        '',
        'CV point metrics average across two repeats of eight trip folds. Decoded CV pools observations from the first eight folds. '
        f'The deployed model’s mean paired point gain over distance + heading is {grouped_gain*100:+.2f} percentage points '
        f'(paired SD {grouped_spread*100:.2f} pp). Training and test metrics use the ordered labels, so the earlier geometric-label percentages are on a different target.',
        f'Rescored on the same ordered labels, the frozen legacy pipeline reaches '
        f'{legacy.baseline_correct/legacy.observations:.2%} before decoding and '
        f'{legacy.decoded_correct/legacy.observations:.2%} after decoding. The comparison measures the complete pipeline changes, including the expanded candidate set.',
        '',
        '## Labels and coverage',
        '',
        f'Ordered alignment labels every observation. The full 80-trip training split contains {c["training_observations"]:,} observations. '
        f'Of those, {c["training_observations_with_candidate"]:,} have the correct arc in the candidate set and enter the model fit. '
        f'The remaining {c["training_candidate_misses"]:,} stay in validation and coverage denominators. '
        f'The ten-trip student test split has {c["student_test_observations"]:,} labels and {c["student_test_candidate_misses"]} candidate misses.',
        '',
        'The largest train candidate-miss counts come from ' + ' and '.join(
            f'`{r.trip_id}` ({r.candidate_misses:,} misses out of {r.observations:,} observations)'
            for r in major.itertuples()) + '. These include extended portions far from the supplied reference. See the per-trip alignment audit.',
        '',
        '## Calibration, tuning, and decoding',
        '',
        f'Optuna ran {len(tuning)} TPE trials ({len(early)} completed, {sum(tuning.state == "PRUNED")} pruned). '
        f'Within completed trials, {stops} stopped tree growth before their cap in at least one fold. '
        f'The training-trip exponential residual mean is {c["fitted_exponential_beta_m"]:.2f} m. '
        f'The selected decoder uses beta {d["beta_m"]:.2f} m, a {d["max_speed_mps"]:.0f} m/s speed cap, '
        f'a {d["max_gap_s"]:.0f} s long-gap threshold, and emission weight {d["emission_weight"]:.2f}. '
        'Large-gap transitions must pass directed routing and the strict speed cap. '
        'The evaluation keeps the student test split out of tuning, model selection, and probability calibration.',
        '',
        '## Student test by trip',
        '',
        *evidence,
        '',
        '## Legacy error audit on trip 00000077',
        '',
        f'The original classifier made {original["classifier"]["errors"]} labeled errors. '
        f'{original["classifier"]["overlapping"]["reverse_arc"]} were exact reverse arcs, '
        f'{original["classifier"]["overlapping"]["stationary_or_slow"]} were stationary or slow, and '
        f'{original["classifier"]["overlapping"]["candidate_miss"]} lacked the legacy target candidate. '
        f'The original decoder had {original["decoder"]["overlapping"]["reverse_arc"]} reverse-arc errors. '
        'Intersection, parallel-road and adjacent-unlabeled tags are overlapping diagnostic proxies. The error-level CSV gives each flag.',
        f'With ordered labels, trip `00000077` reaches {trip_77.classifier_top1:.2%} before decoding and '
        f'{trip_77.decoded_top1:.2%} after. The largest test-trip loss from decoding is '
        f'{abs(largest_drop.decoder_delta)*100:.2f} percentage points on `{largest_drop.trip_id}`.',
        '',
    ]
    (root / 'RESULTS.md').write_text('\n'.join(output), encoding='utf-8')


if __name__ == '__main__':
    main()
