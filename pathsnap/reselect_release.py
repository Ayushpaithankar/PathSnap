"""Refit the learned release scorer using frozen, training-only review evidence."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .grouped_features import CONTEXT_FEATURES, RULES
from .grouped_training import read_split
from .review_experiment import (HGB, NO_CONTEXT, RANKER, UNTUNED, arrays,
                                export, paired_interval, read_quality, train_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prepared', type=Path)
    parser.add_argument('review', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    review = args.review
    cv = pd.read_csv(review / 'cv_summary.csv').set_index('method')
    learned = (UNTUNED, NO_CONTEXT, RANKER, HGB)
    selected = max(learned, key=lambda name: cv.loc[name, 'trusted_decoded_top1'])
    if selected != UNTUNED:
        print('CV-selected learned method:', selected)
    else:
        print('CV-selected learned method: LightGBM untuned')

    protocol = json.loads((review / 'protocol.json').read_text())
    previous = json.loads((review / 'selection.json').read_text())
    decoder = json.loads((review / 'decoder_parameters.json').read_text())
    train = read_split(args.prepared / 'train')
    test = read_split(args.prepared / 'student_test')
    quality = read_quality(args.prepared, 'train', train)
    ids = sorted(set(train) - set(protocol['excluded_trips']))
    assert len(train) == 80 and len(test) == 10 and len(ids) == 78
    assert not set(train) & set(test)
    with threadpool_limits(limits=args.threads):
        matrix = arrays(train, quality, ids, len(CONTEXT_FEATURES))
        model = train_model(selected, matrix, args.threads,
                            json.loads((review / 'ranker_parameters.json').read_text()))

    deployed = review / 'deployed.joblib'
    nearest = review / 'nearest_rule.joblib'
    if not nearest.exists():
        shutil.copy2(deployed, nearest)
    export(selected, model, decoder['slopes'], decoder['configs'],
           previous['transition_beta_m'], deployed, test)
    detail = pd.read_csv(review / 'cv_per_trip.csv', dtype={'trip_id': str})
    selected_vs_nearest = paired_interval(detail, selected, RULES[0],
                                          'trusted_decoded_correct', 'trusted_observations')
    tuned_vs_untuned = paired_interval(detail, RANKER, UNTUNED,
                                      'trusted_decoded_correct', 'trusted_observations')
    previous.update(selected=selected,
        selection_policy='highest trusted decoded assessment CV accuracy among trained models; rules remain baselines',
        trusted_cv_selected_vs_nearest=selected_vs_nearest,
        trusted_cv_tuned_vs_untuned=tuned_vs_untuned)
    (review / 'selection.json').write_text(json.dumps(previous, indent=2) + '\n')
    print(json.dumps(previous, indent=2))


if __name__ == '__main__':
    main()
