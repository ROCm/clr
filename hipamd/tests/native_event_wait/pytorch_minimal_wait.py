# MIT License
# 
# Copyright (C) Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Plain PyTorch counterpart of the shared 2,048-increment wait-only reproducer."""
import csv
import json
import os
from pathlib import Path
import sys
import torch

if torch.cuda.device_count() != 1:
    raise RuntimeError('Run on exactly one allocated GPU')
torch.cuda.init()
libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if 'libamdhip64.so' in line or 'libhsa-runtime64.so' in line})
print(json.dumps({'torch': torch.__version__, 'compiled_hip': torch.version.hip,
                  'enabled': os.environ.get('GPU_NATIVE_EVENT_WAIT', '0'),
                  'libraries': libraries}), file=sys.stderr)
compute, side = torch.cuda.Stream(), torch.cuda.Stream()
value = torch.zeros(1, device='cuda', dtype=torch.int32)
with torch.cuda.stream(compute):
    value.add_(1)
compute.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=compute):
    for _ in range(2048):
        value.add_(1)
begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
writer = csv.writer(sys.stdout)
writer.writerow(['block', 'repeat', 'case', 'gpu_us', 'pending', 'correct'])
cases = ['alone', 'pending_wait', 'ready_wait']
for block in range(-1, 5):
    for offset in range(3):
        case = cases[(offset + block) % 3]
        for repeat in range(8):
            with torch.cuda.stream(compute):
                value.zero_()
            compute.synchronize()
            with torch.cuda.stream(compute):
                begin.record()
                graph.replay()
                end.record()
            pending = 0
            if case == 'pending_wait':
                pending = int(not end.query())
                if not pending:
                    raise RuntimeError('Wait submitted after producer completion')
                side.wait_event(end)
            end.synchronize()
            if case == 'ready_wait':
                side.wait_event(end)
            side.synchronize()
            if value.item() != 2048:
                raise RuntimeError('Incorrect producer output')
            if block >= 0:
                writer.writerow([block, repeat, case, begin.elapsed_time(end) * 1000, pending, 1])
                sys.stdout.flush()
