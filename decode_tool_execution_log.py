# -*- coding: utf-8 -*-
"""decode_tool_execution_log.py -- dump tool_execution_log.pkl as plain text.

Standalone (stdlib only, no simutils import): reads ``tool_execution_log.pkl``
from the current directory and writes a human-readable transcript to
``chat.txt`` in the same directory -- one block per turn, with the LLM
thinking / final response shown as plain text instead of buried in a pickled
dict.

Run from inside the folder that contains tool_execution_log.pkl:
    python decode_tool_execution_log.py
"""

import json
import pickle
from pathlib import Path

INPUT_PKL = "tool_execution_log.pkl"
OUTPUT_TXT = "chat.txt"


def format_value(value):
    """Pretty-print a value for the transcript: try JSON first (the LLM's
    final_response is usually a JSON object as text), fall back to str()."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            return value
    return str(value)


def format_entry(entry, index):
    lines = []
    if isinstance(entry, dict):
        turn = entry.get("turn", index)
        lines.append(f"===== Turn {turn} =====")
        for key, value in entry.items():
            if key == "turn":
                continue
            lines.append(f"\n--- {key} ---")
            lines.append(format_value(value))
    else:
        lines.append(f"===== Entry {index} =====")
        lines.append(str(entry))
    return "\n".join(lines)


def main():
    data = pickle.loads(Path(INPUT_PKL).read_bytes())

    entries = data if isinstance(data, (list, tuple)) else [data]
    blocks = [format_entry(entry, i) for i, entry in enumerate(entries)]

    Path(OUTPUT_TXT).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries from {INPUT_PKL} to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
