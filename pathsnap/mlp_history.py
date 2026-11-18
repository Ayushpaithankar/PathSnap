"""Record a real trip-held-out training/validation curve for the grouped MLP."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .grouped_models import fit_model
from .grouped_training import fitting_arrays, folds, read_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    trips = read_split(args.prepared / 'train')
    assert len(trips) == 80
    valid_ids = folds(trips, 42)[0]
    train_ids = sorted(set(trips) - set(valid_ids))
    train = fitting_arrays(trips, train_ids)
    valid = fitting_arrays(trips, valid_ids)
    model = fit_model('Grouped MLP', *train, validation=valid)
    frame = pd.DataFrame(model.loss_history_)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'grouped_mlp_loss_history.csv', index=False)
    ax = frame.plot(x='epoch', y=['train_grouped_log_loss', 'valid_grouped_log_loss'],
                    marker='o', figsize=(7, 4), grid=True)
    ax.set(title='Grouped MLP: held-out trip loss', xlabel='Epoch', ylabel='Grouped log loss')
    plt.tight_layout()
    plt.savefig(args.out / 'grouped_mlp_loss_history.png', dpi=180)
    plt.close()
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
