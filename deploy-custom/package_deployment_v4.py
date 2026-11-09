#!/usr/bin/env python3
"""Package the Custom34 deployment with generated-image and expert-chunk review support."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bundle_installer', ROOT/'deploy-custom/install_custom34_deployment.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
FILES = (*installer.SERVER_FILES, installer.CLIENT_SOURCE,
         'deploy-custom/install_custom34_deployment.py', 'deploy-custom/check_client_rotation.py',
         'deploy-custom/validate_custom34_bridge.py', 'deploy-custom/README.md',
         'deploy-custom/CHUNK_REVIEW.md', 'deploy-custom/DEPLOYMENT_V4_TESTING.md',
         'deploy-custom/test_deployment_v3.py', 'deploy-custom/test_chunk_review.py',
         'deploy-custom/test_client_server_contract.py', 'deploy-custom/test_custom34_rtc_adapter.py',
         'deploy-custom/test_sync_client_control.py', 'deploy-custom/test_deployment_install.py',
         'tests/test_letterbox_image_preproc.py')


def build(output):
    payloads = {name: (ROOT/name).read_bytes() for name in FILES}
    manifest = dict(version='custom34-deployment-v4', files={
        name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())})
    payloads['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    payloads['README.md'] = payloads['deploy-custom/README.md']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    print(json.dumps(dict(output=str(output.resolve()), bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest(), files=len(payloads)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'deliveries/custom34-deployment-v4.zip')
    build(parser.parse_args().out)
