#!/usr/bin/env python3
"""Install the letterbox image preprocessing patch with backups; keep local configs/train.sh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


SOURCE_FILES = (
    "src/agenticwam/datasets/lerobot/transforms/image.py",
    "src/agenticwam/datasets/lerobot/processors/agenticwam_processor.py",
    "robotwin/deploy/deploy_policy.py",
    "deploy-custom/custom34_server.py",
    "deploy/custom34_server.py",
    "scripts/configure_custom_image_preproc.py",
)


def install(bundle, project):
    bundle, project = Path(bundle).resolve(), Path(project).resolve()
    if not (project / "src/agenticwam").is_dir():
        raise ValueError("Choose the existing agentic-wam project root")
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    payloads = {}
    for name in SOURCE_FILES:
        source, target = bundle / name, project / name
        if not source.resolve().is_relative_to(bundle) or not target.resolve().is_relative_to(project):
            raise ValueError(f"Source/target must stay inside bundle/project: {name}")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Expected a file: {target}")
        payloads[name] = data
    backup = project / ".image-preproc-backups" / str(time.time_ns())
    backup.mkdir(parents=True)
    record = {"replaced": [], "new": []}
    for name in SOURCE_FILES:
        target = project / name
        record["replaced" if target.exists() else "new"].append(name)
        if target.exists():
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    (backup / "installation.json").write_text(json.dumps(record, indent=2) + "\n")
    for name, data in payloads.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            temporary.chmod(mode)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    backup = install(args.bundle_dir, args.project_root)
    print(f"Installed {len(SOURCE_FILES)} letterbox source/tool files. Backup: {backup}")
    print("Local train.sh and YAML configurations were preserved. Restart is required to load the new source.")


if __name__ == "__main__":
    main()
