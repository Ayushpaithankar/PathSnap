"""Charts from saved measured model-comparison and tuning CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path('results/review')
    diagnostics = pd.read_csv(root / 'five_model_diagnostics.csv').set_index('method')
    ordered = pd.read_csv('results/ordered/student_test_comparison.csv').set_index('model')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    diagnostics[['cv_macro_f1','test_macro_f1']].plot.bar(ax=axes[0], color=['#8eafe8','#2457d6'])
    axes[0].set(title='Candidate-row macro F1', xlabel='', ylabel='Macro F1', ylim=(0,1))
    ordered.loc[diagnostics.index, ['classifier_top1','grouped_log_loss']].rename(
        columns={'classifier_top1':'Top-1','grouped_log_loss':'Grouped log loss'}).plot.bar(
        ax=axes[1], color=['#258a75','#d07a38'])
    axes[1].set(title='Student-test score quality', xlabel='', ylabel='Value')
    for ax in axes:ax.tick_params(axis='x', labelrotation=45);ax.grid(axis='y', alpha=.2)
    fig.savefig(root / 'five_model_comparison.png', dpi=180)
    plt.close(fig)

    trials = pd.read_csv(root / 'ranker_optuna_trials.csv')
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(trials.number, trials.value, 'o', color='#8eafe8', label='Trial loss')
    ax.plot(trials.number, trials.value.cummin(), color='#2457d6', label='Best so far')
    ax.set(xlabel='Trial', ylabel='Grouped covered log loss', title='LightGBM Optuna TPE search (18 trials)')
    ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root / 'ranker_optuna_history.png', dpi=180)
    plt.close(fig)

    review = pd.read_csv(root / 'review_macro_f1.csv').pivot(index='method', columns='split', values='macro_f1')
    metrics = pd.read_csv(root / 'test_summary.csv').set_index('method')
    names = ['Nearest road', 'LightGBM untuned', 'LightGBM tuned', 'HistGradientBoosting']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    review.loc[names, ['cv_trusted','student_test']].plot.bar(ax=axes[0], color=['#8eafe8','#2457d6'])
    axes[0].set(title='Review: candidate-row macro F1', xlabel='', ylabel='F1', ylim=(0,1))
    metrics.loc[names, ['point_top1','decoded_top1']].plot.bar(ax=axes[1], color=['#258a75','#d07a38'])
    axes[1].set(title='Student test: point accuracy', xlabel='', ylabel='Accuracy', ylim=(0,1))
    for ax in axes:ax.tick_params(axis='x', labelrotation=45);ax.grid(axis='y', alpha=.2)
    fig.savefig(root / 'review_model_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
