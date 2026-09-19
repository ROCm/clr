"""Spawn-safe launcher: hooks install before workers construct ModelRunner."""

import runpy
from worker_hooks import install

install()

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
