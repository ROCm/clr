"""Capture the retained client's generated inputs before its timing begins."""

import json
import os
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    source = Path(os.environ["NATIVE_ROOT"]) / "phase3/client/benchmark_serving.py"
    sys.path.insert(0, str(source.parent))

    def capture(frame, event, value):
        if (
            event == "return"
            and frame.f_code.co_name == "sample_random_requests"
            and Path(frame.f_code.co_filename) == source
        ):
            sys.setprofile(None)
            tokenizer = frame.f_locals["tokenizer"]
            records = []
            for prompt, input_length, output_length, auxiliary in value:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(token_ids) != input_length:
                    raise AssertionError(
                        "Client token-ID capture disagrees with retained length"
                    )
                records.append(
                    dict(
                        prompt=prompt,
                        input_ids=token_ids,
                        input_length=input_length,
                        output_length=output_length,
                        auxiliary=auxiliary,
                    )
                )
            Path(os.environ["NATIVE_REQUEST_CAPTURE_PATH"]).write_text(
                json.dumps(records)
            )

    sys.setprofile(capture)
    runpy.run_path(str(source), run_name="__main__")
