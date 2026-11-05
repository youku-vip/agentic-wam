#!/usr/bin/env python3
"""Set train/val image resize mode without rewriting other YAML fields."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile
import time

from omegaconf import OmegaConf
import yaml


def mapping_value(node, key):
    if not isinstance(node, yaml.MappingNode) or node.flow_style:
        raise ValueError("Expected ordinary block YAML mappings for data/train/val/processor")
    found = [value for name, value in node.value if name.value == key]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one YAML key: {key}")
    return found[0]


def configure(config_path, mode):
    if mode not in ("stretch", "letterbox"):
        raise ValueError("mode must be stretch or letterbox")
    path = Path(config_path).resolve()
    original = path.read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    tree = yaml.compose(text)
    edits = {}
    data_node = mapping_value(tree, "data")
    for split in ("train", "val"):
        processor = mapping_value(mapping_value(data_node, split), "processor")
        target = mapping_value(processor, "_target_")
        if target.value != "agenticwam.datasets.lerobot.processors.agenticwam_processor.AgenticWAMProcessor":
            raise ValueError(f"data.{split}.processor is not AgenticWAMProcessor")
        modes = [value for name, value in processor.value if name.value == "image_resize_mode"]
        if len(modes) > 1:
            raise ValueError(f"Duplicate data.{split}.processor.image_resize_mode")
        if modes:
            value = modes[0]
            if not isinstance(value, yaml.ScalarNode) or value.style in ("|", ">"):
                raise ValueError("image_resize_mode must be a scalar")
            if value.value != mode:
                edits[(value.start_mark.index, value.end_mark.index)] = mode
        else:
            first_key = processor.value[0][0]
            start = text.rfind("\n", 0, first_key.start_mark.index) + 1
            edits[(start, start)] = " " * first_key.start_mark.column + f"image_resize_mode: {mode}" + newline
    for (start, end), replacement in sorted(edits.items(), reverse=True):
        text = text[:start] + replacement + text[end:]
    before = OmegaConf.to_container(OmegaConf.create(original.decode("utf-8")), resolve=False)
    after = OmegaConf.to_container(OmegaConf.create(text), resolve=False)
    expected = deepcopy(before)
    for split in ("train", "val"):
        expected["data"][split]["processor"]["image_resize_mode"] = mode
    if after != expected:
        raise ValueError("Refusing to change unrelated YAML fields")
    if not edits:
        return None
    backup = path.with_name(path.name + f".image-preproc-{time.time_ns()}.bak")
    shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text.encode("utf-8"))
    try:
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real_post_train/data/custom.yaml"))
    parser.add_argument("--mode", choices=("stretch", "letterbox"), required=True)
    args = parser.parse_args()
    backup = configure(args.config, args.mode)
    print(f"train/val processor.image_resize_mode={args.mode}")
    print(f"Backup: {backup}" if backup else "Already configured; no file changed")
    print("Other YAML fields and train.sh were preserved. Restart training/server to use this recipe.")


if __name__ == "__main__":
    main()
