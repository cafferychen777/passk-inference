"""Reanalyse the real paired counts and verify the paper's dense-grid result."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from passk_inference import __version__, compare, load_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/deepscaler32k'))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    metadata = json.loads((here / 'metadata.json').read_text())
    for entry in metadata['files'].values():
        actual = hashlib.sha256((here / entry['export_file']).read_bytes()).hexdigest()
        if actual != entry['export_sha256']:
            raise ValueError('Example data checksum mismatch')
    result = compare(*load_pair(here / 'base.jsonl', here / 'rl.jsonl'),
                     ks=metadata['analysis']['budget_grid'], bootstrap=4000, alpha=.05, seed=0)
    if (result['prompts'] != 1060 or result['first_loss_interval'] != [11, 61]
            or result['certified_positive_budgets'] != list(range(1, 11))
            or result['certified_negative_budgets'] != list(range(61, 129))):
        raise RuntimeError('Paper result regression: inspect the data and inference before release')
    result['package_version'] = __version__
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'curves.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['k', 'base_pass_at_k', 'rl_pass_at_k', 'rl_minus_base',
                         'simultaneous_lower', 'simultaneous_upper', 'evidence'])
        for i, k in enumerate(result['ks']):
            status = ('gain' if result['lower'][i] > 0 else
                      'loss' if result['upper'][i] < 0 else 'inconclusive')
            writer.writerow([k] + [result[key][i] for key in
                            ['base_pass_at_k', 'rl_pass_at_k', 'mean', 'lower', 'upper']] + [status])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    k = result['ks']
    axes[0].plot(k, result['base_pass_at_k'], label='Base', color='#526574')
    axes[0].plot(k, result['rl_pass_at_k'], label='DeepScaleR', color='#007c83')
    axes[0].set(ylabel='Estimated pass@k', title='Same 1,060 prompts; 128 samples per model')
    axes[0].legend(frameon=False)
    axes[1].fill_between(k, result['lower'], result['upper'], alpha=.22, color='#007c83',
                         label='95% simultaneous band')
    axes[1].plot(k, result['mean'], color='#007c83', label='RL minus base')
    axes[1].axhline(0, color='#526574', lw=.8)
    axes[1].axvspan(11, 61, color='#d99a36', alpha=.12, label='First-loss set: [11, 61]')
    axes[1].set(ylabel='Difference in pass@k', title='32k-token limit; integer budgets 1–128')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set(xlabel='Sampling budget k', xlim=(1, 128))
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.output / 'comparison.png', dpi=180)
    plt.close(fig)
    print('1060 paired prompts | gain: 1–10 | inconclusive: 11–60 | loss: 61–128')
    print('95% first-loss confidence set: [11, 61] (not an exact crossing point)')
    print(f'Wrote result.json, curves.csv, comparison.png to {args.output}')


if __name__ == '__main__':
    main()
