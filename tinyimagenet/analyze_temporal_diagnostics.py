"""Kaggle-only analysis, with explicit completeness and historical provenance."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle, read_json, atomic_json, protocol, jobs, file_hash, digest, now, source_version, config_for


def main():
    require_kaggle()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(ROOT / 'outputs'))
    parser.add_argument('--output', default=str(ROOT / 'analysis'))
    parser.add_argument('--paper-output', default=str(ROOT / 'paper_outputs'))
    parser.add_argument('--include-legacy', action='store_true')
    parser.add_argument('--allow-incomplete', action='store_true', help='Exploratory progress report; cannot produce final archive')
    args = parser.parse_args()
    import pandas as pd
    import numpy as np
    from analysis.legacy import load_legacy
    from analysis.statistics import all_analyses
    from analysis.report import generate_tables, plots, save_table, summary
    inputs, output, paper = Path(args.input), Path(args.output), Path(args.paper_output)
    output.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    frames, performances, resources, hashes, missing = [], [], [], {}, []
    spec, ph = protocol()
    version = source_version()
    for job in jobs():
        run = inputs / job['run_id']
        if not (run / 'completed.json').exists():
            missing.append(job['run_id'])
            continue
        identity = read_json(run / 'completed.json')
        if identity['smoke'] or identity['protocol_hash'] != ph or identity['source_version'] != version or identity['config_hash'] != config_for(job['method'])[2]:
            raise ValueError('Incompatible run: ' + job['run_id'])
        for name in ['completed.json', 'diagnostics/temporal_diagnostics.csv', 'metrics/accuracy_matrix.csv', 'metrics/continual_metrics.csv', 'metrics/task_metrics.csv', 'system/resource_usage.csv']:
            path = run / name
            if not path.exists():
                raise FileNotFoundError('Completed run lacks required artifact: ' + str(path))
            hashes[path.relative_to(inputs).as_posix()] = file_hash(path)
        frame = pd.read_csv(run / 'diagnostics/temporal_diagnostics.csv')
        for field in ['run_id', 'dataset', 'method', 'seed', 'protocol_hash', 'config_hash', 'source_version']:
            if not (frame[field] == identity[field]).all():
                raise ValueError('Diagnostic row provenance mismatch: ' + job['run_id'] + ':' + field)
        from analysis.statistics import METRICS
        if not np.isfinite(frame[METRICS + ['task1_accuracy', 'task1_forgetting']].to_numpy(float)).all():
            raise ValueError('Nonfinite required diagnostic values: ' + job['run_id'])
        matrix = pd.read_csv(run / 'metrics/accuracy_matrix.csv').drop(columns='trained_task').to_numpy(float)
        if matrix.shape != (10, 10) or not np.isfinite(matrix[np.tril_indices(10)]).all():
            raise ValueError('Incomplete accuracy matrix: ' + job['run_id'])
        expected = {(0, 100)} | {(t, e) for t in range(1, 10) for e in range(0, 101, 5)}
        if set(zip(frame.task, frame.epoch)) != expected or len(frame) != len(expected):
            raise ValueError('Missing/duplicate diagnostic checkpoints: ' + job['run_id'])
        frames.append(frame)
        performances.append(identity)
        resources.append(pd.read_csv(run / 'system/resource_usage.csv'))
    if missing and not args.allow_incomplete:
        raise RuntimeError(f'{12 - len(missing)}/12 complete. Resume remaining jobs or use --allow-incomplete for a clearly provisional report.')
    if args.include_legacy:
        older, report = load_legacy(ROOT)
        frames.extend(older)
        save_table(pd.DataFrame(report), output, 'legacy_availability')
    if not frames:
        raise RuntimeError('No measured diagnostics available; no tables or plots generated')
    temporal = pd.concat(frames, ignore_index=True)
    temporal.to_csv(output / 'harmonized_temporal_diagnostics.csv', index=False)
    analyses = all_analyses(temporal, spec)
    for name, frame in analyses.items():
        save_table(frame, output, name)
        if name != 'bootstrap_intervals' and not frame.empty:
            groups = ['dataset', 'method', 'config_hash', 'protocol_hash', 'geometry_metric']
            groups += [c for c in ['horizon', 'variant', 'label'] if c in frame]
            values = [c for c in ['pearson', 'spearman', 'partial_pearson', 'partial_spearman'] if c in frame]
            seed_frame = frame
            if name == 'within_task_correlations':
                seed_frame = frame.groupby(groups + ['seed'], dropna=False)[values].mean().reset_index()
            save_table(summary(seed_frame, groups, values), output, name + '_seed_summary')
    performance = pd.DataFrame(performances)
    tables = generate_tables(performance, analyses, paper)
    figure_manifest = plots(temporal, performance, analyses, pd.concat(resources, ignore_index=True) if resources else pd.DataFrame(), paper / 'figures')
    atomic_json(paper / 'figures_manifest.json', figure_manifest)
    (paper / 'neurocomputing_figures_manifest.md').write_text('\n'.join(f"- {r['figure']}: {r['description']}" for r in figure_manifest) + '\n', encoding='utf-8')
    table_paths = sorted(paper.glob('table_*.tex'))
    (paper / 'neurocomputing_results_tables.tex').write_text('% Requires booktabs. NA denotes an unavailable estimate.\n' + '\n'.join(p.read_text(encoding='utf-8') for p in table_paths), encoding='utf-8')
    a, b, controls, lag = tables
    lines = [f'# Measured empirical summary ({12 - len(missing)}/12 Tiny ImageNet runs)', '',
             'This is an association study. Checkpoints are dependent observations; independent-seed summaries are the principal evidence. No causal or universal claim follows from these estimates.',
             'NC4 records agreement. NC2 and angular drift duplicate the same angle statistic and are not two independent pieces of evidence.',
             'Historical CIFAR centroid references differ from the new validation references. See legacy_availability for missing metrics or horizons.', '']
    for _, row in a.iterrows():
        lines.append(f"{row['method']}: measured average accuracy {row['average_accuracy_mean']:.4f}, sample SD {row['average_accuracy_std']:.4f}, available seeds {row['seeds']}. This is not a superiority test.")
    tiny_raw = analyses['raw_correlations']
    tiny_raw = tiny_raw[tiny_raw.dataset == 'tinyimagenet']
    for _, row in tiny_raw.iterrows():
        lines.append(f"{row.method}, seed {row.seed}, {row.geometry_metric}: Pearson {row.pearson:.4f}, Spearman {row.spearman:.4f}; {row.reason}. Compare the partial and detrended estimates before interpreting the level association.")
    lines.extend(['', 'Weak, negative and unavailable estimates remain in the tables. No metric, seed, method or lag was selected for being favorable. Lagged level association alone does not demonstrate predictive value beyond current forgetting.'])
    (paper / 'neurocomputing_empirical_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(output / 'completion_report.json', {'complete': not missing, 'runs_complete': len(performances), 'missing_runs': missing, 'protocol_hash': ph, 'source_version': version, 'input_hashes': hashes, 'timestamp': now(), 'analyses': list(analyses), 'figures': figure_manifest, 'legacy_included': args.include_legacy, 'input_digest': digest(hashes)})
    print('Analysis written:', output, 'Paper artifacts:', paper, flush=True)


if __name__ == '__main__':
    main()
