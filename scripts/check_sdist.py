"""Check a setuptools source distribution against the reviewed release inventory."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
GENERATED_FILES = {
    'PKG-INFO', 'setup.cfg',
    'passk_inference.egg-info/PKG-INFO',
    'passk_inference.egg-info/SOURCES.txt',
    'passk_inference.egg-info/dependency_links.txt',
    'passk_inference.egg-info/entry_points.txt',
    'passk_inference.egg-info/requires.txt',
    'passk_inference.egg-info/top_level.txt',
}


def check_sdist(archive_path, root=ROOT):
    root = Path(root)
    inventory = (root / 'release-files.txt').read_text(encoding='utf-8').splitlines()
    if not inventory or len(inventory) != len(set(inventory)):
        raise ValueError('Empty or duplicate release paths')
    expected = set(inventory)
    for name in expected:
        path = root / name
        if (PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f'Invalid release path: {name}')
    seen = set()
    prefixes = set()
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or member.name.startswith('/') or '..' in parts:
                raise ValueError(f'Invalid archive path: {member.name}')
            prefixes.add(parts[0])
            if member.isdir():
                continue
            if not member.isfile() or len(parts) < 2:
                raise ValueError(f'Non-regular archive file: {member.name}')
            name = '/'.join(parts[1:])
            if name in seen:
                raise ValueError(f'Duplicate archive file: {name}')
            seen.add(name)
            if name not in expected | GENERATED_FILES:
                raise ValueError(f'Unexpected source-distribution file: {name}')
            if name in expected:
                if archive.extractfile(member).read() != (root / name).read_bytes():
                    raise ValueError(f'Source-distribution content differs: {name}')
    if len(prefixes) != 1:
        raise ValueError('Source distribution must have exactly one top-level directory')
    missing = expected - seen
    if missing:
        raise ValueError(f'Missing source-distribution files: {sorted(missing)}')
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    try:
        count = check_sdist(args.archive)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.error(str(exc))
    print(f'Validated {count} reviewed source files; only known packaging metadata added')


if __name__ == '__main__':
    main()
