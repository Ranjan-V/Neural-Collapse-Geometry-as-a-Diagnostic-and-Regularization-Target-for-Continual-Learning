"""Tables and figures generated strictly from observed seed-level results."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'finetune': 'FT', 'lwf': 'LwF', 'etfs': 'ETF-S', 'lwf_etf': 'LwF+ETF', 'ewc': 'EWC', 'etf_dense_historical': 'ETF dense'}
COLORS = {'finetune': '#576273', 'lwf': '#007C91', 'etfs': '#C34C47', 'lwf_etf': '#6A9548', 'ewc': '#9A61A6', 'etf_dense_historical': '#BC8D31'}


def summary(frame, groups, columns):
    records = []
    if frame.empty:
        return pd.DataFrame(columns=groups + ['n_seeds'])
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        row['n_seeds'] = group.seed.nunique() if 'seed' in group else len(group)
        row['seeds'] = ','.join(map(str, sorted(group.seed.unique()))) if 'seed' in group else ''
        for col in columns:
            if col not in group:
                continue
            values = pd.to_numeric(group[col], errors='coerce')
            row[col + '_mean'] = values.mean()
            row[col + '_std'] = values.std(ddof=1)
            row[col + '_n'] = int(values.notna().sum())
        records.append(row)
    return pd.DataFrame(records)


def save_table(frame, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f'{name}.csv', index=False)
    # No optional tabulate dependency is required for Markdown.
    formatter = lambda v: f'{v:.4f}' if isinstance(v, (float, np.floating)) and np.isfinite(v) else ('NA' if pd.isna(v) else str(v))
    show = frame.map(formatter) if hasattr(frame, 'map') else frame.applymap(formatter)
    markdown = '| ' + ' | '.join(show.columns) + ' |\n| ' + ' | '.join(['---'] * len(show.columns)) + ' |\n'
    markdown += '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in show.itertuples(index=False, name=None))
    (output / f'{name}.md').write_text(markdown + '\n', encoding='utf-8')
    # Pandas escape=True prevents config/method identifiers corrupting LaTeX.
    frame.to_latex(output / f'{name}.tex', index=False, escape=True, na_rep='NA', float_format='%.4f')


def paper_table(frame, output, name, columns):
    """Keep full provenance in CSV; use readable journal-width display columns."""
    frame.to_csv(Path(output) / f'{name}_full_stats.csv', index=False)
    rows = []
    for _, row in frame.iterrows():
        cells = {k: row[k] for k in ['dataset', 'method', 'geometry_metric', 'horizon'] if k in row}
        for metric, label in columns:
            mean, std, count = row.get(metric + '_mean'), row.get(metric + '_std'), row.get(metric + '_n', 0)
            if mean is None or pd.isna(mean):
                cells[label] = 'NA (n=0)'
            elif pd.isna(std):
                cells[label] = f'{mean:.3f} (n={count})'
            else:
                cells[label] = f'{mean:.3f} +/- {std:.3f} (n={count})'
        rows.append(cells)
    save_table(pd.DataFrame(rows), output, name)


def generate_tables(performance, analyses, output):
    group = ['dataset', 'method', 'config_hash', 'protocol_hash']
    a = summary(performance, group, ['average_accuracy', 'average_forgetting', 'backward_transfer'])
    Path(output).mkdir(parents=True, exist_ok=True)
    paper_table(a[a.dataset == 'tinyimagenet'] if 'dataset' in a else a, output, 'table_A_performance', [('average_accuracy', 'Avg accuracy'), ('average_forgetting', 'Forgetting'), ('backward_transfer', 'BWT')])
    stats_group = group + ['geometry_metric']
    b = summary(analyses['raw_correlations'], stats_group, ['pearson', 'spearman'])
    paper_table(b[b.dataset == 'tinyimagenet'] if 'dataset' in b else b, output, 'table_B_diagnostics', [('pearson', 'Pearson'), ('spearman', 'Spearman')])
    c = analyses['raw_correlations'].copy()
    for source, prefix in [('partial_correlations', 'partial'), ('detrended_correlations', 'detrended')]:
        part = analyses[source][stats_group + ['seed', 'pearson']].rename(columns={'pearson': prefix + '_pearson'})
        c = c.merge(part, on=stats_group + ['seed'], how='outer', validate='one_to_one')
    within = analyses['within_task_correlations']
    if not within.empty:
        within = within[within.variant == 'levels'].groupby(stats_group + ['seed'], dropna=False).pearson.mean().rename('within_task_pearson').reset_index()
        c = c.merge(within, on=stats_group + ['seed'], how='outer', validate='one_to_one')
    controls = summary(c, stats_group, ['pearson', 'partial_pearson', 'detrended_pearson', 'within_task_pearson'])
    paper_table(controls, output, 'table_C_time_controls', [('pearson', 'Raw'), ('partial_pearson', 'Partial'), ('detrended_pearson', 'Detrended'), ('within_task_pearson', 'Within task')])
    lag = summary(analyses['lagged_correlations'], stats_group + ['horizon'], ['pearson', 'spearman', 'partial_pearson', 'partial_spearman'])
    paper_table(lag, output, 'table_D_lags', [('pearson', 'Pearson'), ('spearman', 'Spearman'), ('partial_pearson', 'Partial Pearson')])
    paper_table(b, output, 'table_E_cross_dataset', [('pearson', 'Pearson'), ('spearman', 'Spearman')])
    return a, b, controls, lag


def plots(temporal, performance, analyses, resources, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})
    manifest = []
    def save(fig, name, description):
        for ext in ['pdf', 'png']:
            fig.savefig(output / (name + '.' + ext), bbox_inches='tight')
        plt.close(fig)
        manifest.append({'figure': name, 'description': description, 'pdf': name + '.pdf', 'png': name + '.png'})
    tiny = temporal[temporal.dataset == 'tinyimagenet']
    perf = performance[performance.dataset == 'tinyimagenet'] if not performance.empty else performance
    if not perf.empty:
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), layout='constrained')
        for ax, metric, label in zip(axes, ['average_accuracy', 'average_forgetting', 'backward_transfer'], ['Final average accuracy', 'Average forgetting', 'Backward transfer']):
            for i, (method, block) in enumerate(perf.groupby('method')):
                ax.scatter(np.full(len(block), i), block[metric], color=COLORS[method], s=18, zorder=3)
                ax.errorbar(i, block[metric].mean(), yerr=block[metric].std() if len(block) > 1 else None, fmt='_', color='black', capsize=3)
            methods = sorted(perf.method.unique())
            ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=25)
            ax.set_ylabel(label)
        save(fig, '01_performance', 'Individual seeds with mean and sample standard deviation. Partial runs excluded.')
    if not tiny.empty:
        fig, ax = plt.subplots(figsize=(7, 2.8), layout='constrained')
        for (method, seed), data in tiny.groupby(['method', 'seed']):
            ax.plot(data.global_progress, data.task1_forgetting, color=COLORS[method], alpha=0.65, label=f'{LABELS[method]} {seed}', linewidth=0.8)
        ax.set(xlabel='Normalized global progress', ylabel='Task-1 forgetting')
        ax.legend(ncol=4)
        save(fig, '02_forgetting_trajectory', 'Measured validation checkpoints, all method/seed trajectories.')
        for metric, name in [('nc3', '03_nc3'), ('nc4', '04_nc4'), ('angle_drift', '05_angular')]:
            fig, axes = plt.subplots(1, 4, figsize=(7, 2.5), layout='constrained', sharey=True)
            for ax, method in zip(axes, ['finetune', 'lwf', 'etfs', 'lwf_etf']):
                for seed, data in tiny[tiny.method == method].groupby('seed'):
                    ax.scatter(data[metric], data.task1_forgetting, s=5, alpha=0.45, label=str(seed))
                ax.set(title=LABELS[method], xlabel=metric.upper())
            axes[0].set_ylabel('Task-1 forgetting')
            axes[-1].legend()
            save(fig, name, 'Association plot; points are dependent checkpoints, not independent replicates.')
    raw, partial = analyses['raw_correlations'], analyses['partial_correlations']
    if not raw.empty:
        join = ['dataset', 'method', 'seed', 'geometry_metric', 'config_hash', 'protocol_hash']
        combined = raw.merge(partial, on=join, suffixes=('_raw', '_partial'))
        fig, ax = plt.subplots(figsize=(3.5, 3), layout='constrained')
        for method, data in combined[combined.dataset == 'tinyimagenet'].groupby('method'):
            ax.scatter(data.pearson_raw, data.pearson_partial, label=LABELS.get(method, method), color=COLORS.get(method), s=18)
        ax.plot([-1, 1], [-1, 1], ':', color='grey', linewidth=0.7)
        ax.set(xlabel='Raw Pearson r', ylabel='Partial Pearson r', xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
        if ax.collections:
            ax.legend()
        save(fig, '06_raw_partial', 'Fixed valid correlation bounds [-1,1]; every estimable seed/metric.')
        lag = analyses['lagged_correlations']
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), layout='constrained', sharey=True)
        metrics = ['nc1', 'nc2', 'nc3', 'nc4', 'etf_drift', 'angle_drift']
        for ax, horizon in zip(axes, ['5', '10', 'task_end']):
            for i, method in enumerate(['finetune', 'lwf', 'etfs', 'lwf_etf']):
                data = lag[(lag.dataset == 'tinyimagenet') & (lag.horizon == horizon) & (lag.method == method)]
                for j, metric in enumerate(metrics):
                    ax.scatter(np.full(sum(data.geometry_metric == metric), j + (i - 1.5) * 0.13), data.loc[data.geometry_metric == metric, 'pearson'], color=COLORS[method], s=10, label=LABELS[method] if j == 0 else None)
            ax.set(title='+' + horizon if horizon != 'task_end' else 'Task end', ylim=(-1.05, 1.05))
            ax.set_xticks(range(6), metrics, rotation=65)
        axes[0].set_ylabel('Lagged Pearson r')
        axes[-1].legend(loc='best')
        save(fig, '07_lagged', 'Prespecified within-task horizons; absent estimates remain absent.')
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), layout='constrained', sharey=True)
        for ax, dataset in zip(axes, ['cifar10', 'cifar100', 'tinyimagenet']):
            for i, method in enumerate(['finetune', 'lwf', 'etfs', 'lwf_etf']):
                data = raw[(raw.dataset == dataset) & (raw.method == method)]
                for j, metric in enumerate(metrics):
                    values = data.loc[data.geometry_metric == metric, 'pearson']
                    ax.scatter(np.full(len(values), j + (i - 1.5) * 0.13), values, color=COLORS[method], s=10)
            ax.set(title=dataset, ylim=(-1.05, 1.05))
            ax.set_xticks(range(6), metrics, rotation=65)
        axes[0].set_ylabel('Pearson r')
        save(fig, '08_cross_dataset', 'Historical CIFAR and new Tiny ImageNet estimates; centroid reference differences documented.')
        fig, axes = plt.subplots(1, 4, figsize=(7, 2.6), layout='constrained', sharey=True)
        for ax, method in zip(axes, ['finetune', 'lwf', 'etfs', 'lwf_etf']):
            for seed, data in raw[(raw.dataset == 'tinyimagenet') & (raw.method == method)].groupby('seed'):
                values = data.set_index('geometry_metric').reindex(metrics).pearson
                ax.plot(range(6), values, 'o-', markersize=3, linewidth=0.8, label=str(seed))
            ax.set(title=LABELS[method], ylim=(-1.05, 1.05))
            ax.set_xticks(range(6), metrics, rotation=65)
        axes[0].set_ylabel('Pearson r')
        if axes[-1].lines:
            axes[-1].legend()
        save(fig, '10_seed_consistency', 'All available prespecified seeds, including weak or negative estimates.')
    if not resources.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout='constrained')
        for ax, method in zip(axes, ['etfs', 'lwf_etf']):
            for seed, data in resources[resources.method == method].groupby('seed'):
                progress = data.task * 100 + data.epoch
                for metric in ['ce_loss', 'centroid_loss', 'angle_loss']:
                    if metric in data:
                        ax.plot(progress, data[metric], linewidth=0.6, alpha=0.6, label=f'{metric} {seed}')
            ax.set(title=LABELS[method], xlabel='Cumulative epochs', ylabel='Loss')
            ax.set_yscale('symlog', linthresh=0.01)
            if ax.lines:
                ax.legend(ncol=2, fontsize=5)
        save(fig, '09_etf_components', 'Epoch means include scheduled zero anchor steps; symlog shows observed scale without clipping.')
    return manifest
