"""Restore one task-owned writable cache from the frozen local seed."""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

seed = Path("/native-runtime-cache/seed")
root = Path("/native-runtime-cache")
output = Path(sys.argv[1])
assert output.is_relative_to(Path(os.environ["NATIVE_ROOT"]) / "phase4")
output.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
inventory = json.loads((seed / "inventory.json").read_text())
symlinks = json.loads((seed / "symlinks.json").read_text())
for name in ("working-cache", "aiter-jit"):
    destination = root / name
    assert destination.is_dir() and (seed / name).is_dir()
    # Keep the bind-mounted directory inode; only replace task-owned contents.
    for child in destination.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(
        seed / name,
        destination,
        dirs_exist_ok=True,
        symlinks=True,
        copy_function=shutil.copyfile,
    )
for relative, expected in symlinks.items():
    path = root / relative
    if not path.is_symlink() or str(path.readlink()) != expected:
        raise AssertionError(f"Cache seed symlink mismatch: {relative}")
for relative, expected in inventory.items():
    path = root / relative
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise AssertionError(f"Cache seed byte mismatch: {relative}")
(output / "cache-reset.json").write_text(
    json.dumps(
        {
            "seconds": time.monotonic() - start,
            "seed_manifest_sha256": hashlib.sha256(
                (seed / "inventory.json").read_bytes()
            ).hexdigest(),
            "files_verified": len(inventory),
            "symlinks_verified": len(symlinks),
            "valid": True,
        },
        indent=2,
    )
)
