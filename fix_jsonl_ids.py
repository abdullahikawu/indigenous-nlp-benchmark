"""Convert JSONL record IDs to sequential integers in place."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


INPUT_PATH = Path("data/nup/raw/raw_data_group_01.jsonl")


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    output_fd, output_name = tempfile.mkstemp(
        prefix="raw_data_group_01_",
        suffix=".jsonl",
        dir=INPUT_PATH.parent,
        text=True,
    )

    record_count = 0
    try:
        with INPUT_PATH.open("r", encoding="utf-8") as source, os.fdopen(
            output_fd, "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} must contain a JSON object."
                    )

                record_count += 1
                record["id"] = record_count
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(output_name, INPUT_PATH)
    except Exception:
        # The original input remains untouched if validation or writing fails.
        try:
            os.unlink(output_name)
        except FileNotFoundError:
            pass
        raise

    print(f"Updated {record_count} records in {INPUT_PATH}")
    print("IDs now run from 1 to", record_count)


if __name__ == "__main__":
    main()
