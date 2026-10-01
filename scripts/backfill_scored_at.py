"""One-off: set `scored_at` on every leaderboard record from the date git first added the file.

Run from the repo root, then `mia-evals leaderboard` to re-render:
    python scripts/backfill_scored_at.py
"""

import json
import subprocess
from pathlib import Path


def main():
    for path in sorted(Path("leaderboard").glob("*/records/*.json")):
        d = json.loads(path.read_text())
        if d.get("scored_at"):
            continue
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%aI", "--", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert out, f"{path} has no git history; commit it first or set scored_at by hand"
        d["scored_at"] = out[-1]
        path.write_text(json.dumps(d, indent=2, sort_keys=False) + "\n")
        print(path.name, out[-1])


if __name__ == "__main__":
    main()
