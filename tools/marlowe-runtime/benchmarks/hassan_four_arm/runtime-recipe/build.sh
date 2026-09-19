#!/usr/bin/env bash
# Build the exact release implementation, independently of the caller's checkout.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
: "${RUNTIME_WORK_DIR:?Set RUNTIME_WORK_DIR to a new build directory}"
: "${RUNTIME_BUILD_BASE_IMAGE:?Set the immutable build image identity from source-lock.json}"
export RUNTIME_BUILD_BASE_IMAGE
: "${ROCM_PATH:=/opt/rocm}"
: "${RUNTIME_BUILD_JOBS:=8}"
mkdir -p "$RUNTIME_WORK_DIR"
work=$(cd "$RUNTIME_WORK_DIR" && pwd)
for project in clr hip; do
 repo=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]+"_repository"])' "$recipe/source-lock.json" "$project")
 revision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]+"_commit"])' "$recipe/source-lock.json" "$project")
 if [[ -e "$work/$project" ]]; then echo "Use a fresh build directory: $work/$project exists" >&2; exit 2; fi
 git init -q "$work/$project"
 git -C "$work/$project" fetch --depth=1 "$repo" "$revision"
 git -C "$work/$project" checkout --detach FETCH_HEAD
 test "$(git -C "$work/$project" rev-parse HEAD)" = "$revision"
done
python3 -m pip install --no-build-isolation --require-hashes --target "$work/python-deps" -r "$recipe/requirements-build.txt"
export PYTHONPATH="$work/python-deps${PYTHONPATH:+:$PYTHONPATH}"
cmake -S "$work/clr" -B "$work/build" \
 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
 -D__HIP_ENABLE_PCH=OFF -DHIP_COMMON_DIR="$work/hip" \
 -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX="$work/install" \
 -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
 -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++"
cmake --build "$work/build" -j "$RUNTIME_BUILD_JOBS"
python3 "$recipe/assemble.py" "$work" "$recipe"
