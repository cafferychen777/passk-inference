"""Independent combinatorial checks and inference boundary contracts."""

import itertools
import json

import numpy as np
import pytest

from passk_inference import (certify, compare, fit_kernel, kernel_curve,
                             load_pair, pass_at_k, simultaneous_band)


def test_passk_matches_exhaustive_sampling_without_replacement():
    for n in range(1, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                draws = list(itertools.combinations(range(n), k))
                expected = np.mean([any(i < c for i in draw) for draw in draws])
                assert pass_at_k([c], [n], k)[0] == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("c,n,k", [([-1], [4], 1), ([5], [4], 1),
    ([1.2], [4], 1), ([1], [0], 1), ([1], [4], 5), ([1], [4], True),
    ([1], [4], 1.5), ([np.nan], [4], 1), ([], [], 1)])
def test_bad_counts_fail(c, n, k):
    with pytest.raises(ValueError):
        pass_at_k(c, n, k)


def test_crossing_requires_ordered_two_signs():
    assert certify([1, 2], [.1, -.3], [.2, -.1])["certified_crossing"]
    assert not certify([1, 2], [-.3, .1], [-.1, .2])["certified_crossing"]
    assert not certify([1, 2], [-.3, -.2], [-.1, -.1])["certified_crossing"]


def test_sparse_grid_does_not_claim_unobserved_integer_interval():
    result = certify([1, 8, 64], [.1, .1, -.3], [.2, .2, -.1])
    assert result["certified_crossing"]
    assert result["first_loss_interval"] is None
    assert certify([1, 2, 3], [.1, 0, -.3], [.2, .2, -.1])["first_loss_interval"] == [2, 3]


def test_identical_pair_and_degenerate_columns():
    result = compare([0, 2, 4], 4, [0, 2, 4], 4, bootstrap=99)
    assert not result["certified_crossing"]
    assert result["mean"] == [0.0] * 4
    assert result["degenerate_columns"] == [0, 1, 2, 3]


def test_band_reproduces_pilot_multiplier_reference():
    rows = np.random.default_rng(41).normal(size=(30, 5))
    result = simultaneous_band(rows, bootstrap=257, seed=9)
    mean = rows.mean(axis=0)
    se = rows.std(axis=0, ddof=1) / np.sqrt(30)
    e = np.random.default_rng(9).standard_normal((257, 30))
    q = np.quantile(np.max(np.abs(e @ (rows - mean)) / (30 * se), axis=1), .95)
    np.testing.assert_allclose(result["lower"], mean - q * se)


def test_kernel_curve_matches_single_component_bernoulli():
    fit = {"W": [[1.0]], "pg": [.2], "qg": [.4]}
    np.testing.assert_allclose(kernel_curve(fit, [1, 2, 8]),
                               [.8**k - .6**k for k in [1, 2, 8]])


def test_kernel_fit_likelihood_matches_returned_weights():
    from scipy.stats import binom
    c, d = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 3])
    fit = fit_kernel(c, 4, d, 4, Lp=8, Lq=8, iters=21)
    p, q = np.clip(fit["pg"], 1e-12, 1-1e-12), np.clip(fit["qg"], 1e-12, 1-1e-12)
    likelihood = sum(np.log(np.sum(fit["W"] * binom.pmf(a, 4, p)[:, None]
                                        * binom.pmf(b, 4, q)[None, :])) for a, b in zip(c, d))
    assert fit["loglik"] == pytest.approx(likelihood)
    assert fit["W"].sum() == pytest.approx(1)
    assert isinstance(fit["converged"], bool)


def test_loader_aligns_and_rejects_duplicates_and_missing_prompts(tmp_path):
    base, rl = tmp_path / "b.jsonl", tmp_path / "r.jsonl"
    a, b = {"id": "a", "c": 1, "n": 4}, {"id": "b", "c": 3, "n": 4}
    base.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    rl.write_text(json.dumps(b) + "\n" + json.dumps(a) + "\n")
    assert load_pair(base, rl)[0] == load_pair(base, rl)[2]
    rl.write_text(json.dumps(a) + "\n" + json.dumps(a) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_pair(base, rl)
    rl.write_text(json.dumps(a) + "\n")
    with pytest.raises(ValueError, match="IDs differ"):
        load_pair(base, rl)


def test_cli_emits_finite_json(tmp_path, capsys):
    from passk_inference.cli import main
    path = tmp_path / "counts.jsonl"
    path.write_text('{"id":"a","c":1,"n":4}\n{"id":"b","c":3,"n":4}\n')
    assert main(["--base", str(path), "--rl", str(path), "--bootstrap", "19"]) == 0
    assert json.loads(capsys.readouterr().out)["prompts"] == 2


def test_rare_success_with_large_generation_budget_stays_resolved():
    assert pass_at_k([1], [10**9], 1)[0] == pytest.approx(1e-9, rel=1e-12)
    assert pass_at_k([1], [10**9], 100)[0] == pytest.approx(1e-7, rel=1e-12)


def test_constant_prompt_differences_do_not_certify_population_crossing():
    result = compare([1] * 5, 4, [2] * 5, 4, bootstrap=99)
    assert result["certified_positive_budgets"] == []
    assert result["lower"] == [-1.0] * 4
    assert result["upper"] == [1.0] * 4


def test_loader_order_does_not_change_seeded_inference(tmp_path):
    base, rl = tmp_path / "base.jsonl", tmp_path / "rl.jsonl"
    rows = [{"id": str(i), "c": i, "n": 4} for i in range(4)]
    def write(path, values):
        path.write_text("".join(json.dumps(r) + "\n" for r in values))
    write(base, rows)
    write(rl, [{**r, "c": 4-r["c"]} for r in rows])
    first = compare(*load_pair(base, rl), bootstrap=99)
    write(base, rows[::-1])
    second = compare(*load_pair(base, rl), bootstrap=99)
    assert first == second


def test_difference_direction_and_reported_curves():
    a = compare([0, 1, 3, 4], 4, [1, 3, 2, 4], 4, bootstrap=257)
    b = compare([1, 3, 2, 4], 4, [0, 1, 3, 4], 4, bootstrap=257)
    np.testing.assert_allclose(np.array(a['rl_pass_at_k']) - a['base_pass_at_k'], a['mean'])
    np.testing.assert_allclose(a['lower'], -np.array(b['upper']))
    np.testing.assert_allclose(a['upper'], -np.array(b['lower']))
    assert a['difference_direction'] == 'rl_minus_base'
    assert sorted(a['certified_positive_budgets'] + a['certified_negative_budgets']
                  + a['inconclusive_budgets']) == a['ks']


def test_real_paper_dense_grid_reference():
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / 'examples/deepscaler32k'
    reference = json.loads((folder / 'expected.json').read_text())
    result = compare(*load_pair(folder / 'base.jsonl', folder / 'rl.jsonl'))
    assert result['first_loss_interval'] == [11, 61]
    assert result['certified_positive_budgets'] == list(range(1, 11))
    assert result['certified_negative_budgets'] == list(range(61, 129))
    for key in ['mean', 'lower', 'upper', 'base_pass_at_k', 'rl_pass_at_k']:
        np.testing.assert_allclose(result[key], reference[key], atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('field', ['c', 'n'])
@pytest.mark.parametrize('value', [True, False, '4', 1.0, 1.5, None])
def test_jsonl_rejects_non_integer_count_types(tmp_path, field, value):
    from passk_inference import load_counts
    path = tmp_path / 'invalid.jsonl'
    row = {'id': 'a', 'c': 1, 'n': 4, field: value}
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match=rf':1: {field} must be a JSON integer'):
        load_counts(path)


def test_file_provenance_hashes_the_parsed_snapshot(tmp_path, monkeypatch):
    import hashlib
    import passk_inference.io as count_io
    from passk_inference import __version__, compare_files
    base, rl = tmp_path / 'base.jsonl', tmp_path / 'rl.jsonl'
    raw = b'{"id":"a","c":1,"n":4}\n{"id":"b","c":2,"n":4}\n'
    base.write_bytes(raw)
    rl.write_bytes(raw)
    original = count_io._parse_counts
    def mutate_after_snapshot(content, source):
        base.write_text('changed after the snapshot')
        return original(content, source)
    monkeypatch.setattr(count_io, '_parse_counts', mutate_after_snapshot)
    result = compare_files(base, rl, bootstrap=19)
    assert result['package_version'] == __version__
    assert result['result_format_version'] == '1.0'
    assert result['mean'] == [0.0] * 4
    assert result['input_provenance']['files'] == {
        role: {'sha256': hashlib.sha256(raw).hexdigest()} for role in ['base', 'rl']}
    assert str(tmp_path) not in json.dumps(result)


def test_cli_report_json_csv_and_plot_agree(tmp_path, capsys):
    import csv
    pytest.importorskip('matplotlib')
    from passk_inference.cli import main
    base, rl = tmp_path / 'base.jsonl', tmp_path / 'rl.jsonl'
    base.write_text('{"id":"a","c":0,"n":4}\n{"id":"b","c":2,"n":4}\n')
    rl.write_text('{"id":"a","c":1,"n":4}\n{"id":"b","c":3,"n":4}\n')
    output = tmp_path / 'report'
    main(['--base', str(base), '--rl', str(rl), '--bootstrap', '19', '--output', str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / 'result.json').read_text())
    with (output / 'curves.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        assert float(row['rl_minus_base']) == result['mean'][i]
        assert float(row['simultaneous_lower']) == result['lower'][i]
        assert float(row['simultaneous_upper']) == result['upper'][i]
        expected = ('gain' if result['lower'][i] > 0 else
                    'loss' if result['upper'][i] < 0 else 'inconclusive')
        assert row['evidence'] == expected
    assert (output / 'comparison.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


@pytest.mark.parametrize('ks,expected', [([1, 3], 'sparse grid'), ([1], '[1, ∞)')])
def test_report_handles_sparse_and_unbounded_sets(tmp_path, monkeypatch, ks, expected):
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure
    from passk_inference import write_report
    captured = []
    monkeypatch.setattr(Figure, 'savefig', lambda fig, *args, **kwargs:
                        captured.append([ax.get_title() for ax in fig.axes]))
    result = compare([0, 1, 2, 3], 4, [1, 2, 3, 4], 4, ks=ks, alpha=.1, bootstrap=99)
    write_report(result, tmp_path)
    assert expected in captured[0][1]
    if len(ks) == 1:
        assert '90%' in captured[0][1]


def test_json_cli_does_not_import_plotting(tmp_path, capsys, monkeypatch):
    import builtins
    from passk_inference.cli import main
    original = builtins.__import__
    def no_plotting(name, *args, **kwargs):
        if name.startswith('matplotlib'):
            raise ImportError('plotting dependency absent')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_plotting)
    path = tmp_path / 'counts.jsonl'
    path.write_text('{"id":"a","c":1,"n":4}\n{"id":"b","c":2,"n":4}\n')
    assert main(['--base', str(path), '--rl', str(path), '--bootstrap', '19']) == 0
    assert json.loads(capsys.readouterr().out)['prompts'] == 2
    with pytest.raises(SystemExit) as exc:
        main(['--base', str(path), '--rl', str(path), '--output', str(tmp_path / 'report')])
    assert exc.value.code == 2
    assert 'Plotting requires matplotlib' in capsys.readouterr().err
