"""Source-release boundary regressions independent of build backend behavior."""
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location(
    'check_sdist', Path(__file__).resolve().parents[1] / 'scripts/check_sdist.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize('problem', [None, 'extra', 'missing', 'altered', 'symlink', 'duplicate'])
def test_sdist_inventory_boundary(tmp_path, problem):
    (tmp_path / 'README.md').write_text('Public\n')
    (tmp_path / 'release-files.txt').write_text('README.md\nrelease-files.txt\n')
    entries = [('README.md', b'Public\n'), ('release-files.txt', b'README.md\nrelease-files.txt\n'),
               ('PKG-INFO', b'Name: example\n')]
    if problem == 'extra':
        entries.append(('docs/unlisted-marker.md', b'Private draft\n'))
    elif problem == 'missing':
        entries.pop(0)
    elif problem == 'altered':
        entries[0] = ('README.md', b'Changed\n')
    elif problem == 'duplicate':
        entries.append(entries[0])
    archive_path = tmp_path / 'example.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, content in entries:
            info = tarfile.TarInfo('example-1.0/' + name)
            if problem == 'symlink' and name == 'README.md':
                info.type = tarfile.SYMTYPE
                info.linkname = 'outside'
                archive.addfile(info)
            else:
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    if problem is None:
        assert checker.check_sdist(archive_path, tmp_path) == 2
    else:
        with pytest.raises(ValueError):
            checker.check_sdist(archive_path, tmp_path)


@pytest.mark.parametrize('problem', [None, 'delayed', 'hash_mismatch', 'missing'])
def test_pypi_distribution_identity(tmp_path, monkeypatch, problem):
    import hashlib
    import json

    spec = importlib.util.spec_from_file_location(
        'verify_pypi', Path(__file__).resolve().parents[1] / 'scripts/verify_pypi.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    entries = []
    for suffix in ('-py3-none-any.whl', '.tar.gz'):
        path = tmp_path / ('passk_inference-1.2.3' + suffix)
        path.write_bytes(suffix.encode())
        entries.append({'filename': path.name,
                        'digests': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    calls = []

    def response(url, timeout):
        calls.append(url)
        assert url == 'https://pypi.org/pypi/passk-inference/1.2.3/json'
        visible = entries
        if problem == 'missing' or (problem == 'delayed' and len(calls) == 1):
            visible = entries[:1]
        if problem == 'hash_mismatch':
            visible = [dict(entries[0], digests={'sha256': '0' * 64}), entries[1]]
        return io.StringIO(json.dumps({'urls': visible}))

    monkeypatch.setattr(verifier, 'urlopen', response)
    monkeypatch.setattr(verifier.time, 'sleep', lambda delay: None)
    if problem == 'hash_mismatch':
        with pytest.raises(ValueError, match='hash differs'):
            verifier.verify('1.2.3', tmp_path, attempts=2)
        assert len(calls) == 1
    elif problem == 'missing':
        with pytest.raises(RuntimeError, match='retry limit'):
            verifier.verify('1.2.3', tmp_path, attempts=2)
        assert len(calls) == 2
    else:
        verifier.verify('1.2.3', tmp_path, attempts=2)
        assert len(calls) == (2 if problem == 'delayed' else 1)
