"""Build a source release using an exact reviewed path allowlist."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = (ROOT / 'release-files.txt').read_text().splitlines()
    if not files or len(files) != len(set(files)):
        raise ValueError('Empty or duplicate release paths')
    version = re.search(r'__version__ = "([^"]+)"',
                        (ROOT / 'passk_inference/_version.py').read_text())[1]
    payload = {}
    for name in files:
        path = ROOT / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.is_file() or not path.resolve().is_relative_to(ROOT)):
            raise ValueError(f'Invalid release path: {name}')
        raw = path.read_bytes()
        if path.suffix != '.png':
            text = raw.decode('utf-8')
            # Avoid including local paths, private hosts or common token formats.
            patterns = [r'/Users/\w+', r'/home/\w+', r'gh[pousr]_[A-Za-z0-9]{20,}',
                        r'github_pat_[A-Za-z0-9_]{20,}', r'-----' + r'BEGIN .*PRIVATE' + r' KEY-----']
            if any(re.search(pattern, text) for pattern in patterns):
                raise ValueError(f'Potential private information in {name}')
        payload[name] = raw
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()}
    payload['SHA256SUMS.json'] = (json.dumps(hashes, indent=2, sort_keys=True) + '\n').encode()
    output = ROOT / 'dist' / f'passk-inference-{version}-source.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(f'passk-inference-{version}/{name}', (2026, 9, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    print(f'{output.name}: {len(payload)} files, {output.stat().st_size} bytes')
    print(f'SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
