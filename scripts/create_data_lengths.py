"""Generate a JSON file containing bundled data table lengths."""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "sts2rl" / "data" / "json"
OUTPUT_FILE = DATA_DIR / "json_lengths.json"


def main() -> None:
    """Write json_lengths.json for all bundled JSON data tables."""
    lengths = {}

    for file_path in sorted(DATA_DIR.glob("*.json")):
        if file_path == OUTPUT_FILE:
            continue

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            lengths[file_path.name] = len(data)
        else:
            lengths[file_path.name] = None

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(lengths, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
