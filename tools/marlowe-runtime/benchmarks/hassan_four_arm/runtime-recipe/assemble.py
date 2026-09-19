"""Package the pinned userspace runtime and licenses without modifying the host."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

work, recipe = [Path(p).resolve() for p in sys.argv[1:]]
lock = json.loads((recipe / 'source-lock.json').read_text())
if os.environ.get('RUNTIME_BUILD_BASE_IMAGE') != lock['build_base_image']:
    raise SystemExit('Build orchestrator must supply the pinned RUNTIME_BUILD_BASE_IMAGE identity')
for project in ['clr', 'hip']:
    actual = subprocess.check_output(['git', '-C', str(work / project), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != lock[project + '_commit']:
        raise SystemExit(f'Unexpected {project} source: {actual}')
    dirty = subprocess.check_output(['git', '-C', str(work / project), 'status', '--porcelain'], text=True)
    if dirty:
        raise SystemExit(f'Dirty {project} source')
rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
hip = work / 'build/hipamd/lib' / lock['hip_library']
hsa = (rocm / 'lib/libhsa-runtime64.so').resolve(strict=True)
if hsa.name != lock['hsa_library'] or hashlib.sha256(hsa.read_bytes()).hexdigest() != lock['hsa_sha256']:
    raise SystemExit(f'Unqualified HSA runtime: {hsa}')
out = work / 'dist' / lock['release']
if out.exists():
    raise SystemExit(f'Refusing to replace existing release {out}')
(out / 'lib').mkdir(parents=True)
(out / 'licenses').mkdir()
manifest = dict(lock, libraries={}, aliases={}, build_tools={})
for library, aliases in [(hip, ['libamdhip64.so', 'libamdhip64.so.7']),
                         (hsa, ['libhsa-runtime64.so', 'libhsa-runtime64.so.1'])]:
    destination = out / 'lib' / library.name
    shutil.copy2(library, destination)
    manifest['libraries'][library.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    for alias in aliases:
        (out / 'lib' / alias).symlink_to(destination.name)
        manifest['aliases'][alias] = destination.name
for name in ['run', 'verify.py']:
    shutil.copy2(recipe / name, out / name)
for project in ['clr', 'hip']:
    shutil.copy2(work / project / 'LICENSE.md', out / 'licenses' / (project + '-LICENSE.md'))
shutil.copy2(recipe / 'ROCr-LICENSE.txt', out / 'licenses/ROCr-LICENSE.txt')
for command in [['cmake', '--version'], [str(rocm / 'llvm/bin/clang'), '--version']]:
    manifest['build_tools'][command[0]] = subprocess.check_output(command, text=True)
manifest['recipe_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in recipe.iterdir() if p.is_file()}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = shutil.make_archive(str(out), 'gztar', root_dir=out.parent, base_dir=out.name)
print(json.dumps({'directory': str(out), 'archive': archive,
                  'sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest()}, indent=2))
