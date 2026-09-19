"""Strict ROCm SMI query parser, retained from the audited cleanup helper."""

import re

def parse_health(health: str) -> dict:
    def device_values(label: str) -> dict:
        pairs = re.findall(r"GPU\[(\d+)\].*" + re.escape(label) + r": (\d+)", health)
        assert len(pairs) == 8, f"Incomplete/duplicate device query: {label}"
        values = {int(device): int(value) for device, value in pairs}
        assert set(values) == set(range(8)), f"Wrong device census: {label}"
        return values

    allocated = device_values("GPU Memory Allocated (VRAM%)")
    busy = device_values("GPU use (%)")
    total = device_values("VRAM Total Memory (B)")
    used = device_values("VRAM Total Used Memory (B)")
    assert all(
        0 <= used[device] <= total[device] and total[device] > 0 for device in total
    ), "Invalid raw memory query"
    assert (
        health.count("KFD Processes") == 1 and health.count("End of ROCm SMI Log") == 1
    ), "Missing/multiple KFD section"
    section = health.split("KFD Processes", 1)[1].split("End of ROCm SMI Log", 1)[0]
    rows = []
    saw_info = False
    saw_header = False
    explicit_empty = False
    for line in section.splitlines():
        text = line.strip()
        if not text or set(text) == {"="}:
            continue
        if text == "KFD process information:":
            assert not saw_info, "Repeated KFD info marker"
            saw_info = True
        elif (
            " ".join(text.split())
            == "PID PROCESS NAME GPU(s) VRAM USED SDMA USED CU OCCUPANCY"
        ):
            assert not saw_header, "Repeated KFD header"
            saw_header = True
        elif text in {"No KFD PIDs currently running", "No KFD PIDs found"}:
            assert not explicit_empty, "Repeated empty KFD marker"
            explicit_empty = True
        else:
            match = re.fullmatch(r"(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)", text)
            assert (
                match is not None and saw_info and saw_header and not explicit_empty
            ), f"Unparsed KFD content: {text}"
            pid, name, devices, vram, sdma, occupancy = match.groups()
            rows.append(
                dict(
                    pid=int(pid),
                    name=name,
                    devices=devices,
                    vram_bytes=int(vram),
                    sdma_bytes=int(sdma),
                    occupancy=occupancy,
                )
            )
    assert (rows and saw_info and saw_header and not explicit_empty) or (
        explicit_empty and not rows
    ), "Incomplete KFD query"
    assert len({row["pid"] for row in rows}) == len(rows), "Duplicate KFD process"
    return dict(
        gpu_allocated_vram_percent=allocated,
        gpu_busy_percent=busy,
        gpu_total_vram_bytes=total,
        gpu_used_vram_bytes=used,
        all_reported_raw_vram_bytes_zero=all(value == 0 for value in used.values()),
        kfd_processes=rows,
        kfd_explicit_empty=explicit_empty,
    )
