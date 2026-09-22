"""Render a comparison without changing or recomputing its statistical results."""

import csv
import json
from pathlib import Path


def write_report(result, output, *, base_label='Base', rl_label='RL'):
    """Write result.json, curves.csv and comparison.png; requires the report extra.

    Existing files with these names are replaced. The figure uses the reported
    confidence level and grid, including sparse grids and unbounded first loss.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ValueError("Plotting requires matplotlib; install '.[report]' from the repository") from exc
    # Serialize before writing, so invalid/nonfinite JSON never becomes an artifact.
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    ks = result['ks']
    confidence = f"{100 * (1 - result['alpha']):g}%"
    fig = Figure(figsize=(10, 4), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    sparse = result['first_loss_scope'] != 'all_integers_1_to_K'
    style = {'marker': 'o', 'linestyle': 'none'} if sparse else {}
    axes[0].plot(ks, result['base_pass_at_k'], label=base_label, color='#526574', **style)
    axes[0].plot(ks, result['rl_pass_at_k'], label=rl_label, color='#007c83', **style)
    axes[0].set(ylabel='Estimated pass@k', title=f"Same {result['prompts']:,} paired prompts")
    axes[0].legend(frameon=False)
    if sparse:
        error = [[m - lo for m, lo in zip(result['mean'], result['lower'])],
                 [hi - m for m, hi in zip(result['mean'], result['upper'])]]
        axes[1].errorbar(ks, result['mean'], yerr=error, fmt='o', capsize=3,
                         color='#007c83', label=f'{confidence} simultaneous bounds (listed k)')
    else:
        axes[1].fill_between(ks, result['lower'], result['upper'], color='#007c83', alpha=.22,
                             label=f'{confidence} simultaneous band')
        axes[1].plot(ks, result['mean'], color='#007c83', label=f'{rl_label} minus {base_label}')
    axes[1].axhline(0, color='#526574', lw=.8)
    interval = result['first_loss_interval']
    if interval is None:
        title = 'First-loss set unavailable on sparse grid'
    elif interval[1] is None:
        title = f"{confidence} first-loss set: [{interval[0]}, ∞)"
    else:
        title = f"{confidence} first-loss set: [{interval[0]}, {interval[1]}]"
        axes[1].axvspan(*interval, color='#d99a36', alpha=.12)
    axes[1].set(ylabel='Difference in pass@k', title=title)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set(xlabel='Sampling budget k',
               xlim=(ks[0], ks[-1]) if len(ks) > 1 else (ks[0] - .5, ks[0] + .5))
        ax.spines[['top', 'right']].set_visible(False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ('result.json', 'curves.csv', 'comparison.png')}
    fig.savefig(paths['comparison.png'], dpi=180)
    paths['result.json'].write_text(serialized, encoding='utf-8')
    with paths['curves.csv'].open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['k', 'base_pass_at_k', 'rl_pass_at_k', 'rl_minus_base',
                         'simultaneous_lower', 'simultaneous_upper', 'evidence'])
        for i, k in enumerate(ks):
            evidence = ('gain' if result['lower'][i] > 0 else
                        'loss' if result['upper'][i] < 0 else 'inconclusive')
            writer.writerow([k] + [result[key][i] for key in
                            ('base_pass_at_k', 'rl_pass_at_k', 'mean', 'lower', 'upper')] + [evidence])
    return paths
