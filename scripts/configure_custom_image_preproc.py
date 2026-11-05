#!/usr/bin/env python3
"""Set train/val image geometry without rewriting unrelated YAML fields."""
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


def configure(config_path, mode, size=None, scale_up=None):
    if mode not in ("stretch", "letterbox"):
        raise ValueError("mode must be stretch or letterbox")
    if size is not None:
        if len(size) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 or int(v) % 32 for v in size):
            raise ValueError("size must contain two positive multiples of 32: height width")
        size = [int(v) for v in size]
    if scale_up is not None and not isinstance(scale_up, bool):
        raise ValueError("scale_up must be a boolean")
    path = Path(config_path).resolve()
    original = path.read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    tree = yaml.compose(text)
    edits = {}
    insertions = set()
    before = OmegaConf.to_container(OmegaConf.create(text), resolve=False)
    expected = deepcopy(before)

    def set_scalar(node, value):
        if not isinstance(node, yaml.ScalarNode) or node.style in ("|", ">"):
            raise ValueError("Expected a simple YAML scalar")
        if node.value != value:
            edits[(node.start_mark.index, node.end_mark.index)] = value

    def set_field(node, name, value):
        if not isinstance(node, yaml.MappingNode) or node.flow_style:
            raise ValueError("Expected a block YAML mapping")
        fields = [item for key, item in node.value if key.value == name]
        if len(fields) > 1:
            raise ValueError(f"Duplicate YAML field: {name}")
        if fields:
            set_scalar(fields[0], value)
        else:
            first_key = node.value[0][0]
            start = text.rfind("\n", 0, first_key.start_mark.index) + 1
            if (start, name) not in insertions:
                edits[(start, start)] = edits.get((start, start), "") + " " * first_key.start_mark.column + f"{name}: {value}" + newline
                insertions.add((start, name))

    def set_sequence(node, values):
        if not isinstance(node, yaml.SequenceNode) or len(node.value) != len(values):
            raise ValueError("Image sizes must be explicit YAML sequences")
        for child, value in zip(node.value, values):
            set_scalar(child, str(value))

    def set_image_shapes(node, expected_meta):
        images = mapping_value(node, "images")
        if not isinstance(images, yaml.SequenceNode):
            raise ValueError("shape_meta.images must be a list")
        for image, expected_image in zip(images.value, expected_meta["images"]):
            shape = mapping_value(image, "shape")
            if not isinstance(shape, yaml.SequenceNode) or shape.value[0].value != "3":
                raise ValueError("Expected RGB image shape=[3, height, width]")
            set_sequence(shape, [3, *size])
            expected_image["shape"] = [3, *size]

    def set_transform_sizes(node, expected_transforms):
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                set_transform_sizes(value, expected_transforms[key.value])
            return
        if not isinstance(node, yaml.SequenceNode):
            raise ValueError("Image transforms must be explicit lists, optionally mapped by camera")
        found = False
        for transform, expected_transform in zip(node.value, expected_transforms):
            if not isinstance(transform, yaml.MappingNode):
                raise ValueError("Image transform entries must be mappings")
            target = mapping_value(transform, "_target_").value
            if target in ("torchvision.transforms.Resize", "agenticwam.datasets.lerobot.transforms.image.LetterboxResize"):
                set_sequence(mapping_value(transform, "size"), size)
                expected_transform["size"] = list(size)
                if target.endswith("LetterboxResize") and scale_up is not None:
                    set_field(transform, "scale_up", str(scale_up).lower())
                    expected_transform["scale_up"] = scale_up
                found = True
        if not found:
            raise ValueError("Each image transform list must contain an explicit Resize")

    data_node = mapping_value(tree, "data")
    for split in ("train", "val"):
        split_node = mapping_value(data_node, split)
        processor = mapping_value(split_node, "processor")
        expected_split = expected["data"][split]
        expected_processor = expected_split["processor"]
        target = mapping_value(processor, "_target_")
        if target.value != "agenticwam.datasets.lerobot.processors.agenticwam_processor.AgenticWAMProcessor":
            raise ValueError(f"data.{split}.processor is not AgenticWAMProcessor")
        set_field(processor, "image_resize_mode", mode)
        expected_processor["image_resize_mode"] = mode
        if scale_up is not None:
            set_field(processor, "image_letterbox_scale_up", str(scale_up).lower())
            expected_processor["image_letterbox_scale_up"] = scale_up
        if size is not None:
            set_sequence(mapping_value(split_node, "video_size"), size)
            expected_split["video_size"] = list(size)
            set_image_shapes(mapping_value(split_node, "shape_meta"), expected_split["shape_meta"])
            processor_meta = mapping_value(processor, "shape_meta")
            if isinstance(processor_meta, yaml.MappingNode):
                set_image_shapes(processor_meta, expected_processor["shape_meta"])
            for phase in ("train_transforms", "val_transforms"):
                set_transform_sizes(mapping_value(processor, phase), expected_processor[phase])
    for (start, end), replacement in sorted(edits.items(), reverse=True):
        text = text[:start] + replacement + text[end:]
    configured = OmegaConf.create(text)
    after = OmegaConf.to_container(configured, resolve=False)
    if after != expected:
        raise ValueError("Refusing to change unrelated YAML fields")
    if size is not None:
        for split in (configured.data.train, configured.data.val):
            for image in split.processor.shape_meta.images:
                if list(image.shape) != [3, *size]:
                    raise ValueError("Processor shape_meta reference does not resolve to the requested image size")
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
    parser.add_argument("--size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    upscale = parser.add_mutually_exclusive_group()
    upscale.add_argument("--no-upscale", dest="scale_up", action="store_false")
    upscale.add_argument("--upscale", dest="scale_up", action="store_true")
    parser.set_defaults(scale_up=None)
    args = parser.parse_args()
    backup = configure(args.config, args.mode, args.size, args.scale_up)
    print(f"train/val processor.image_resize_mode={args.mode}")
    if args.size is not None:
        print(f"train/val video_size={args.size}; output image shapes and Resize sizes synchronized")
    if args.scale_up is not None:
        print(f"train/val processor.image_letterbox_scale_up={args.scale_up}")
    print(f"Backup: {backup}" if backup else "Already configured; no file changed")
    print("Other YAML fields and train.sh were preserved. Restart training/server to use this recipe.")


if __name__ == "__main__":
    main()
