#!/usr/bin/env python3
"""Install the reviewed deployment bundle; back up each changed file before replacing it."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


SERVER_FILES = (
    'deploy-custom/custom34_server.py',
    'deploy-custom/review_custom34_chunks.py',
    'robotwin/deploy/deploy_policy.py',
    'src/agenticwam/utils/deployment_text_cache.py',
    'src/agenticwam/utils/custom34_review.py',
    'src/agenticwam/utils/review_vae_decode.py',
    'src/agenticwam/datasets/lerobot/transforms/custom_unified_eef.py',
    'src/agenticwam/datasets/lerobot/transforms/image.py',
    'src/agenticwam/datasets/lerobot/processors/agenticwam_processor.py',
    'src/agenticwam/datasets/lerobot/v3/episode_reader.py',
)
CLIENT_SOURCE = 'deploy-custom/rtc_client_agenticwam.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.custom34-', dir=target.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def find_client():
    # Resolve the module location without importing environment or connecting to a robot.
    package = PathFinder.find_spec('environment', [str(Path.cwd()), *sys.path])
    if package is None or not package.submodule_search_locations:
        raise ValueError('cannot locate environment; run from the robot project or pass --client-root')
    module = PathFinder.find_spec('environment.rtc_client_agenticwam', package.submodule_search_locations)
    if module is None or not module.origin:
        raise ValueError('cannot locate environment.rtc_client_agenticwam; pass --client-root')
    return Path(module.origin).resolve()


def install(package_root, server_root=None, client_file=None, dry_run=False):
    package_root = Path(package_root).resolve()
    manifest = json.loads((package_root / 'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        source = package_root / name
        if not source.resolve().is_relative_to(package_root) or not source.is_file() or digest(source) != sha:
            raise ValueError(f'package integrity check failed: {name}')
    pairs = []
    if server_root is not None:
        server_root = Path(server_root).resolve()
        if not (server_root / 'src/agenticwam').is_dir() or not (server_root / 'robotwin').is_dir():
            raise ValueError(f'not an agentic-wam repository root: {server_root}')
        for name in SERVER_FILES:
            target = server_root / name
            if not target.resolve().is_relative_to(server_root):
                raise ValueError(f'symlink destination leaves repository: {target}')
            pairs.append((package_root / name, target))
    if client_file is not None:
        target = Path(client_file).resolve()
        if not target.parent.is_dir():
            raise ValueError(f'client environment directory missing: {target.parent}')
        pairs.append((package_root / CLIENT_SOURCE, target))
    if not pairs:
        raise ValueError('select --server-root, --client-root, --client-file or --client-auto')
    changes = [(source, target) for source, target in pairs if not target.is_file() or digest(source) != digest(target)]
    for source, target in changes:
        print(f'{"WOULD INSTALL" if dry_run else "INSTALL"} {source.relative_to(package_root)} -> {target}')
    if not changes or dry_run:
        return None
    backup_parent = server_root if server_root is not None else Path(client_file).resolve().parent.parent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = backup_parent / 'deployment_backups' / f'{manifest["version"]}-{stamp}-{os.urandom(3).hex()}'
    backup.mkdir(parents=True)
    records = []
    # Finish every backup before the first replacement.
    for index, (source, target) in enumerate(changes):
        previous = backup / f'{index:03d}.original'
        existed = target.is_file()
        if existed:
            shutil.copy2(target, previous)
        records.append(dict(target=str(target), backup=str(previous) if existed else None,
                            before_sha256=digest(target) if existed else None, after_sha256=digest(source)))
    report = backup / 'manifest.json'
    report.write_text(json.dumps(dict(version=manifest['version'], files=records), indent=2) + '\n')
    written = []
    try:
        for (source, target), record in zip(changes, records):
            replace_file(source, target)
            written.append(record)
    except Exception:
        for record in reversed(written):
            target = Path(record['target'])
            if target.is_file() and digest(target) == record['after_sha256']:
                if record['backup']:
                    replace_file(Path(record['backup']), target)
                else:
                    target.unlink()
        print(f'INSTALL FAILED; backups retained at {report}', file=sys.stderr)
        raise
    print(f'BACKUP {report}')
    return report


def undo(report_path):
    records = json.loads(Path(report_path).read_text())['files']
    # Refuse to overwrite edits made after installation; check all files first.
    for record in records:
        target = Path(record['target'])
        if not target.is_file() or digest(target) != record['after_sha256']:
            raise ValueError(f'changed since installation, restore refused: {target}')
        if record['backup'] and digest(Path(record['backup'])) != record['before_sha256']:
            raise ValueError(f'backup integrity check failed: {record["backup"]}')
    for record in records:
        target = Path(record['target'])
        if record['backup']:
            replace_file(Path(record['backup']), target)
        else:
            target.unlink()
        print(f'RESTORED {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--server-root', type=Path)
    clients = parser.add_mutually_exclusive_group()
    clients.add_argument('--client-root', type=Path, help='robot project containing environment/')
    clients.add_argument('--client-file', type=Path, help='exact environment/rtc_client_agenticwam.py path')
    clients.add_argument('--client-auto', action='store_true', help='locate existing module without importing it')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--undo', type=Path, help='restore from the printed backup manifest')
    args = parser.parse_args()
    if args.undo:
        if args.server_root or args.client_root or args.client_file or args.client_auto or args.dry_run:
            parser.error('--undo is used alone')
        undo(args.undo)
        return
    client_file = args.client_file
    if args.client_root:
        client_file = args.client_root / 'environment/rtc_client_agenticwam.py'
    elif args.client_auto:
        client_file = find_client()
    install(args.package_root, args.server_root, client_file, args.dry_run)


if __name__ == '__main__':
    main()
