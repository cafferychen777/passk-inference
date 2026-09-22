"""Reanalyse the real paired counts and verify the paper's dense-grid result."""
import argparse
import hashlib
import json
from pathlib import Path

from passk_inference import compare_files, write_report


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
    result = compare_files(here / 'base.jsonl', here / 'rl.jsonl',
                           ks=metadata['analysis']['budget_grid'], bootstrap=4000, alpha=.05, seed=0)
    if (result['prompts'] != 1060 or result['first_loss_interval'] != [11, 61]
            or result['certified_positive_budgets'] != list(range(1, 11))
            or result['certified_negative_budgets'] != list(range(61, 129))):
        raise RuntimeError('Paper result regression: inspect the data and inference before release')
    write_report(result, args.output, rl_label='DeepScaleR')
    print('1060 paired prompts | gain: 1–10 | inconclusive: 11–60 | loss: 61–128')
    print('95% first-loss confidence set: [11, 61] (not an exact crossing point)')
    print(f'Wrote result.json, curves.csv, comparison.png to {args.output}')


if __name__ == '__main__':
    main()
