"""Verify that PyPI serves the exact built wheel and sdist before installation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def verify(version, directory, *, attempts=18, delay=10):
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Expected a stable MAJOR.MINOR.PATCH version')
    files = sorted(Path(directory).glob('*.whl')) + sorted(Path(directory).glob('*.tar.gz'))
    expected_names = {f'passk_inference-{version}-py3-none-any.whl',
                      f'passk_inference-{version}.tar.gz'}
    if {path.name for path in files} != expected_names:
        raise ValueError('Expected exactly one wheel and one sdist for this version')
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for attempt in range(attempts):
        try:
            with urlopen(f'https://pypi.org/pypi/passk-inference/{version}/json', timeout=20) as response:
                data = json.load(response)
            actual = {entry['filename']: entry['digests']['sha256'] for entry in data['urls']}
            if any(name in actual and actual[name] != digest for name, digest in expected.items()):
                raise ValueError('PyPI file hash differs from the reviewed build')
            if actual == expected:
                print(f'PyPI {version}: wheel and sdist SHA-256 verified')
                return
        except HTTPError as exc:
            if exc.code not in (404, 502, 503, 504):
                raise
        except (URLError, TimeoutError):
            pass
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise RuntimeError('PyPI did not expose both matching distributions before the retry limit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    args = parser.parse_args()
    verify(args.version, args.dist)
