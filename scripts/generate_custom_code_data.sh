#!/usr/bin/env bash
# Download tools when needed, generate labels, and verify the written cache.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/algorithm/agentic-wam}"
PY="${PY:-/usr/local/bin/python3.11}"
CLEAN_ROOT="${CLEAN_ROOT:-/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2}"
OUT_DIR="${OUT_DIR:-$CLEAN_ROOT/coding_v4_regenerated_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
CONFIRM_CURATED_BOUNDARIES="${CONFIRM_CURATED_BOUNDARIES:-1}"
KNOB_AXIS="${KNOB_AXIS:-auto}"
KNOB_ANGLE_SIGN="${KNOB_ANGLE_SIGN:-1}"
CODEGEN_PACKAGE_URL="${CODEGEN_PACKAGE_URL:-https://github.com/youku-vip/agentic-wam/raw/09f23b52ce20131e7890f11ad1fe1effa8dcd10a/deliveries/custom-code-generator-v2.zip}"
CODEGEN_PACKAGE_SHA256="49d43f44fd9b02a248f7d88c28bc91e8bcc31173a8fc316a1440fb7d143eacf6"

fail() { printf '%s\n' "$*" >&2; exit 1; }
has_generator() {
    [[ -f "$1/scripts/generate_custom_code_cache.py" &&
       -f "$1/build_v3_gripper_code_fixed.py" &&
       -f "$1/scripts/rebuild_semantic_code_cache.py" ]]
}

command -v "$PY" >/dev/null || fail "Python 不可用：$PY"
[[ -d "$PROJECT_DIR/src" ]] || fail "项目 src 目录不存在：$PROJECT_DIR/src"
[[ -f "$CLEAN_ROOT/meta/info.json" ]] || fail "数据集元数据不存在：$CLEAN_ROOT/meta/info.json"
[[ ! -e "$OUT_DIR" && ! -L "$OUT_DIR" ]] || fail "输出目录已存在，请通过 OUT_DIR 指定新目录：$OUT_DIR"
[[ "$CONFIRM_CURATED_BOUNDARIES" == 0 || "$CONFIRM_CURATED_BOUNDARIES" == 1 ]] || fail "CONFIRM_CURATED_BOUNDARIES 只能为 0 或 1"

BUNDLED_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${TOOL_DIR:-}" ]]; then
    if has_generator "$BUNDLED_ROOT"; then
        TOOL_DIR="$BUNDLED_ROOT"
    elif has_generator "$PROJECT_DIR"; then
        TOOL_DIR="$PROJECT_DIR"
    else
        TOOL_DIR="$PROJECT_DIR/custom_code_generator_v2"
    fi
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wam-codegen.XXXXXXXX")"
trap 'rm -rf -- "$WORK_DIR"' EXIT
cd -- "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""

"$PY" - "$CLEAN_ROOT" "$WORK_DIR/input_hashes.json" "$CONFIRM_CURATED_BOUNDARIES" <<'PY'
import hashlib, importlib, json, sys
from pathlib import Path
for module in ("numpy", "torch", "pandas", "pyarrow"):
    importlib.import_module(module)
root, destination, confirm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
if info.get("codebase_version") != "v3.0":
    raise SystemExit("输入必须是 LeRobot v3.0 数据集")
if confirm:
    importlib.import_module("agenticwam.datasets.lerobot.v3.episode_reader")
    if not (root / "curation_source_map.json").is_file() or not info.get("trajectory_curation", {}).get("source_root"):
        raise SystemExit("缺少清洗映射或 source_root；仅使用保留数据时设置 CONFIRM_CURATED_BOUNDARIES=0")
names = ("meta/info.json", "curation_source_map.json", "dataset_stats_eef64.json")
destination.write_text(json.dumps({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in names if (root / name).is_file()}), encoding="utf-8")
PY

if ! has_generator "$TOOL_DIR"; then
    [[ ! -e "$TOOL_DIR" && ! -L "$TOOL_DIR" ]] || fail "工具目录不完整，请指定新的 TOOL_DIR：$TOOL_DIR"
    printf '[setup] 安装代码生成工具到 %s\n' "$TOOL_DIR"
    "$PY" - "$CODEGEN_PACKAGE_URL" "$CODEGEN_PACKAGE_SHA256" "$TOOL_DIR" <<'PY'
import hashlib, io, json, os, shutil, sys, tempfile, urllib.request, zipfile
from pathlib import Path
url, expected, target = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
with urllib.request.urlopen(url, timeout=60) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("工具包 SHA256 不匹配；未安装任何文件")
target.parent.mkdir(parents=True, exist_ok=True)
stage = Path(tempfile.mkdtemp(prefix=".wam-codegen-install-", dir=target.parent))
try:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if not (stage / name).resolve().is_relative_to(stage):
                raise ValueError("工具包包含非法路径")
        archive.extractall(stage)
    manifest = json.loads((stage / "MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        content = (stage / entry["path"]).read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("工具文件校验失败：" + entry["path"])
    if target.exists() or target.is_symlink():
        raise ValueError("工具目录已存在，拒绝覆盖")
    os.rename(stage, target)
except BaseException:
    shutil.rmtree(stage)
    raise
PY
fi

printf '[codegen] 数据集：%s\n' "$CLEAN_ROOT"
printf '[codegen] 输出目录：%s\n' "$OUT_DIR"
ARGS=(--root "$CLEAN_ROOT" --out-dir "$OUT_DIR" --knob-axis "$KNOB_AXIS" --knob-angle-sign "$KNOB_ANGLE_SIGN")
if [[ "$CONFIRM_CURATED_BOUNDARIES" == 1 ]]; then
    ARGS+=(--confirm-curation-boundaries)
fi
"$PY" "$TOOL_DIR/scripts/generate_custom_code_cache.py" "${ARGS[@]}"

"$PY" - "$CLEAN_ROOT" "$OUT_DIR" "$TOOL_DIR" "$WORK_DIR/input_hashes.json" <<'PY'
from collections import Counter
import hashlib, io, json, sys, tokenize
from pathlib import Path
import torch
root, out, tool, fingerprints = map(Path, sys.argv[1:])
sys.path.insert(0, str(tool / "scripts"))
from rebuild_semantic_code_cache import calls, validate_inputs
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
order = json.loads((out / "episode_order.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (out / "gripper_code_audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
cache = torch.load(out / "policy_code_idx_custom.pt", map_location="cpu", weights_only=True)
report = json.loads((out / "code_generation_report.json").read_text(encoding="utf-8"))
validate_inputs(cache, order, records, info)
counts = Counter(name for row in records for span in row["spans"] for name, _, _ in calls(span["code"]))
if (report["episodes"] != info["total_episodes"] or report["frames"] != info["total_frames"]
        or Path(report["dataset_root"]).resolve() != root.resolve() or dict(counts) != report["calls"]):
    raise ValueError("生成报告与数据集/缓存不一致")
for code in cache["codes"]:
    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline)):
        raise ValueError("训练代码中仍存在注释")
for name, digest in json.loads(fingerprints.read_text(encoding="utf-8")).items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError("输入元数据或统计文件发生改变：" + name)
print(json.dumps(dict(saved_cache_checks="passed", episodes=report["episodes"], frames=report["frames"],
    calls=dict(counts), task2_without_rotate=report["task2_without_rotate"],
    unresolved_terminal_candidates=report["unresolved_terminal_candidates"],
    refined_rotation_windows=len(report["refined_rotation_windows"]),
    boundary_open_confirmations=len(report["boundary_open_confirmations"])), ensure_ascii=False, indent=2))
print("训练启动参数：")
print("data.dataset_root=" + str(root.resolve()))
print("data.code_cache=" + str((out / "policy_code_idx_custom.pt").resolve()))
if (root / "dataset_stats_eef64.json").is_file():
    print("归一化统计继续使用：" + str((root / "dataset_stats_eef64.json").resolve()))
print("生成报告：" + str((out / "code_generation_report.json").resolve()))
PY
