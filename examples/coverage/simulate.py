"""CPU simulation of simultaneous coverage and crossing detection under known truths."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from passk_inference import __version__, compare

SCENARIOS = {
    'homogeneous_null': {'weights': [1.0], 'base': [.1], 'rl': [.1]},
    'rare_null': {'weights': [1.0], 'base': [.001], 'rl': [.001]},
    'heterogeneous_null': {'weights': [.5, .5], 'base': [.001, .4], 'rl': [.001, .4]},
    'heterogeneous_crossing': {'weights': [.7, .3], 'base': [.15, .02], 'rl': [.3, .0002]},
}


def population_difference(scenario, ks):
    """Exact mixture expectation, not the truth conditional on sampled prompts."""
    ks = np.asarray(ks)
    p, q = np.asarray(scenario['base']), np.asarray(scenario['rl'])
    base = -np.expm1(np.log1p(-p[:, None]) * ks)
    rl = -np.expm1(np.log1p(-q[:, None]) * ks)
    return np.asarray(scenario['weights']) @ (rl - base)


def wilson(successes, repetitions):
    """95% binomial Wilson interval for Monte Carlo uncertainty."""
    z = 1.959963984540054
    rate = successes / repetitions
    denominator = 1 + z * z / repetitions
    center = (rate + z * z / (2 * repetitions)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / repetitions + z * z / (4 * repetitions**2)) / denominator
    return [0.0 if successes == 0 else max(0.0, float(center - radius)),
            1.0 if successes == repetitions else min(1.0, float(center + radius))]


def run_cell(name, prompts, *, repetitions, trials, maximum_k, bootstrap, alpha, seed):
    scenario = SCENARIOS[name]
    scenario_index = list(SCENARIOS).index(name)
    ks = np.arange(1, maximum_k + 1)
    truth = population_difference(scenario, ks)
    gain = np.flatnonzero(truth > 0)
    loss = np.flatnonzero(truth < 0)
    true_crossing = bool(len(gain) and len(loss) and gain[0] < loss[-1])
    first_loss = int(ks[loss[0]]) if len(loss) else None
    covered = detected = first_loss_covered = degenerate = 0
    for repetition in range(repetitions):
        # Data and bootstrap streams are independent and stable across cell order.
        data_seed, band_seed = np.random.SeedSequence(
            [seed, scenario_index, prompts, trials, maximum_k, repetition]).spawn(2)
        rng = np.random.default_rng(data_seed)
        component = rng.choice(len(scenario['weights']), size=prompts, p=scenario['weights'])
        c_base = rng.binomial(trials, np.asarray(scenario['base'])[component])
        c_rl = rng.binomial(trials, np.asarray(scenario['rl'])[component])
        result = compare(c_base, trials, c_rl, trials, ks=ks, bootstrap=bootstrap,
                         alpha=alpha, seed=int(band_seed.generate_state(1)[0]))
        covered += bool(np.all((np.asarray(result['lower']) <= truth)
                               & (truth <= np.asarray(result['upper']))))
        detected += result['certified_crossing']
        lo, hi = result['first_loss_interval']
        # None denotes no loss on the tested grid, not absence of loss forever.
        first_loss_covered += (hi is None if first_loss is None else
                              lo <= first_loss and (hi is None or first_loss <= hi))
        degenerate += bool(result['degenerate_columns'])
    return {
        'scenario': name, 'prompts': prompts, 'true_crossing_on_grid': true_crossing,
        'true_first_loss_on_grid': first_loss, 'true_difference': truth.tolist(),
        'simultaneous_coverage': covered / repetitions,
        'simultaneous_coverage_mc95': wilson(covered, repetitions),
        'crossing_rate_kind': 'power' if true_crossing else 'false_crossing_rate',
        'crossing_rate': detected / repetitions,
        'crossing_rate_mc95': wilson(detected, repetitions),
        'first_loss_set_coverage': first_loss_covered / repetitions,
        'first_loss_set_coverage_mc95': wilson(first_loss_covered, repetitions),
        'degenerate_repetition_fraction': degenerate / repetitions,
    }


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', nargs='+', type=positive, default=[50, 200])
    parser.add_argument('--repetitions', type=positive, default=100)
    parser.add_argument('--bootstrap', type=positive, default=399)
    parser.add_argument('--trials', type=positive, default=64)
    parser.add_argument('--maximum-k', type=positive, default=32)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('output/coverage'))
    args = parser.parse_args()
    if (min(args.prompts) < 2 or args.bootstrap < 2 or args.seed < 0
            or args.maximum_k > args.trials or not 0 < args.alpha < 1):
        parser.error('require prompts >= 2, bootstrap >= 2, seed >= 0, maximum-k <= trials, 0 < alpha < 1')
    cells = []
    for name in SCENARIOS:
        for m in sorted(set(args.prompts)):
            cell = run_cell(name, m, repetitions=args.repetitions, trials=args.trials,
                            maximum_k=args.maximum_k, bootstrap=args.bootstrap,
                            alpha=args.alpha, seed=args.seed)
            cells.append(cell)
            print(f"{name:25s} m={m:4d}: coverage={cell['simultaneous_coverage']:.3f}, "
                  f"{cell['crossing_rate_kind']}={cell['crossing_rate']:.3f}", flush=True)
    output = {
        'simulation_format_version': '1.0', 'package_version': __version__,
        'numpy_version': np.__version__, 'synthetic': True,
        'estimand': 'population mixture mean RL-minus-base pass@k on the full declared grid',
        'config': {key: value for key, value in vars(args).items() if key != 'output'},
        'scenarios': SCENARIOS, 'results': cells,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'simulation.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    with (args.output / 'summary.csv').open('w', newline='') as handle:
        fields = [key for key in cells[0] if key != 'true_difference']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cells)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for name in SCENARIOS:
        group = [cell for cell in cells if cell['scenario'] == name]
        for ax, metric in zip(axes, ['simultaneous_coverage', 'crossing_rate']):
            y = np.array([cell[metric] for cell in group])
            bounds = np.array([cell[metric + '_mc95'] for cell in group])
            label = name.replace('_', ' ')
            if metric == 'crossing_rate':
                label += ' (' + group[0]['crossing_rate_kind'].replace('_', ' ') + ')'
            ax.errorbar([cell['prompts'] for cell in group], y,
                        yerr=np.maximum(0, np.stack([y - bounds[:, 0], bounds[:, 1] - y])),
                        marker='o', capsize=3, label=label)
    axes[0].axhline(1 - args.alpha, color='black', linestyle='--', lw=.8, label='Nominal coverage')
    axes[0].set(ylabel='Whole-grid coverage', title='Known population truth inside the entire band')
    axes[1].set(ylabel='Crossing detection rate', title='Power on crossing; false detection on nulls')
    for ax in axes:
        ax.set(xlabel='Number of independent prompts', ylim=(-.04, 1.04))
        ax.legend(frameon=False, fontsize=7, loc='best')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Synthetic simulation: {args.repetitions} repetitions; bars are 95% Monte Carlo intervals')
    fig.savefig(args.output / 'coverage.png', dpi=180)
    print(f'Wrote simulation.json, summary.csv and coverage.png to {args.output}')


if __name__ == '__main__':
    main()
