"""Installer integrity, backup and rollback checks in temporary folders only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load('install_custom34_deployment')
rotation_check = load('check_client_rotation')


def package(tmp_path):
    root = tmp_path / 'bundle'
    files = {}
    for name in (*installer.SERVER_FILES, installer.CLIENT_SOURCE):
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'new code: {name}')
        files[name] = installer.digest(source)
    (root / 'manifest.json').write_text(json.dumps(dict(version='test', files=files)))
    return root


def repository(tmp_path):
    root = tmp_path / 'target'
    (root / 'src/agenticwam').mkdir(parents=True)
    (root / 'robotwin').mkdir()
    target = root / installer.SERVER_FILES[0]
    target.parent.mkdir()
    target.write_text('original deployment code')
    (root / 'train.sh').write_text('KEEP train.sh')
    return root, target


def test_install_undo_preserves_unrelated_files(tmp_path):
    source = package(tmp_path)
    root, existing = repository(tmp_path)
    report = installer.install(source, server_root=root)
    assert existing.read_text().startswith('new code')
    assert (root / 'train.sh').read_text() == 'KEEP train.sh'
    installer.undo(report)
    assert existing.read_text() == 'original deployment code'
    assert not (root / installer.SERVER_FILES[1]).exists()


def test_tampered_package_is_rejected_before_any_write(tmp_path):
    source = package(tmp_path)
    root, existing = repository(tmp_path)
    (source / installer.SERVER_FILES[-1]).write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        installer.install(source, server_root=root)
    assert existing.read_text() == 'original deployment code'
    assert not (root / 'deployment_backups').exists()


def test_dry_run_and_reinstall_do_not_create_backups(tmp_path):
    source = package(tmp_path)
    root, existing = repository(tmp_path)
    assert installer.install(source, server_root=root, dry_run=True) is None
    assert not (root / 'deployment_backups').exists()
    installer.install(source, server_root=root)
    before = set((root / 'deployment_backups').iterdir())
    assert installer.install(source, server_root=root) is None
    assert set((root / 'deployment_backups').iterdir()) == before


def test_undo_refuses_later_user_edits(tmp_path):
    source = package(tmp_path)
    root, existing = repository(tmp_path)
    report = installer.install(source, server_root=root)
    existing.write_text('later user edit')
    with pytest.raises(ValueError, match='changed since installation'):
        installer.undo(report)
    assert existing.read_text() == 'later user edit'


def test_client_auto_location_does_not_import_robot_package(tmp_path, monkeypatch):
    env = tmp_path / 'environment'
    env.mkdir()
    (env / '__init__.py').write_text('raise RuntimeError("must not import hardware")')
    module = env / 'rtc_client_agenticwam.py'
    module.write_text('raise RuntimeError("must not connect robot")')
    monkeypatch.chdir(tmp_path)
    assert installer.find_client() == module


def test_failed_replacement_rolls_back_prior_writes(tmp_path, monkeypatch):
    source = package(tmp_path)
    root, existing = repository(tmp_path)
    original = installer.replace_file
    calls = []
    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError('simulated filesystem failure')
        original(source, target)
    monkeypatch.setattr(installer, 'replace_file', fail_second)
    with pytest.raises(OSError, match='simulated'):
        installer.install(source, server_root=root)
    assert existing.read_text() == 'original deployment code'
    assert not (root / installer.SERVER_FILES[1]).exists()


@pytest.mark.parametrize('rows,unit,order', [(False, False, 'xyz'), (False, True, 'XYZ'), (True, False, 'xyz')])
def test_sdk_read_only_check_distinguishes_encoding_and_units(rows, unit, order):
    def encode(euler):
        matrix = Rotation.from_euler(order, euler, degrees=unit).as_matrix()
        return matrix[:2].reshape(6) if rows else np.r_[matrix[:, 0], matrix[:, 1]]
    def decode(value):
        pair = np.asarray(value).reshape(2, 3)
        matrix = np.stack([pair[0], pair[1], np.cross(pair[0], pair[1])], axis=0 if rows else 1)
        return Rotation.from_matrix(matrix).as_euler(order, degrees=unit)
    report = rotation_check.check_conventions(SimpleNamespace(euler_angle_to_rot6d=encode, rot6d_to_euler_angle=decode))
    assert report['best_match']['encoding'] == ('first_two_rows' if rows else 'first_two_columns')
    assert report['best_match']['angle_unit'] == ('degree' if unit else 'radian')
    assert report['first_two_columns_verified'] is not rows
    assert report['sdk_pair_verified'] and report['robot_commands_sent'] == 0
