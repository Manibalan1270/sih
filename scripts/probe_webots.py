"""Measure how many AMRs this host's Webots sustains, and record it.

    py scripts/probe_webots.py                 # visual30, the 30-robot world
    py scripts/probe_webots.py --scenario bench3
    py scripts/probe_webots.py --rendered      # with the 3D view, as a demo would run

Launches Webots in batch mode on the scenario's world with the supervisor in
probe mode (see webots/controllers/fleet_supervisor). The supervisor runs the
fleet for a few seconds, measures the real-time factor, writes
config/webots_capacity.json and quits. core.scenarios.clamped reads that file
to shrink a Webots scenario to what the host can show. An unprobed host is
deliberately not guessed at.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import scenarios  # noqa: E402
from scripts import generate_world  # noqa: E402

CANDIDATES = (
    Path(os.environ.get("WEBOTS_HOME", "")) / "msys64" / "mingw64" / "bin" / "webots-bin.exe",
    Path(r"C:\Program Files\Webots\msys64\mingw64\bin\webots-bin.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Webots" / "msys64" / "mingw64" / "bin" / "webots-bin.exe",
    Path("/usr/local/webots/webots"),
    Path("/Applications/Webots.app/Contents/MacOS/webots"),
)


def find_webots() -> Path | None:
    for candidate in CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("webots")
    return Path(found) if found else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Webots capacity.")
    parser.add_argument("--scenario", default="visual30", choices=sorted(scenarios.SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rendered", action="store_true", help="keep the 3D view on during the probe")
    parser.add_argument("--screenshot", type=Path, default=None, help="export the 3D view to this file (implies --rendered)")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)

    webots = find_webots()
    if webots is None:
        print("Webots not found; set WEBOTS_HOME or install it from cyberbotics.com", file=sys.stderr)
        return 2

    scenario = scenarios.get(args.scenario)
    world = generate_world.WORLDS_DIR / f"{scenario.name}.wbt"
    if not world.exists():
        generate_world.main([scenario.name, "--seed", str(args.seed)])

    env = dict(os.environ, ROBOTON_PROBE="1")
    if args.screenshot:
        env["ROBOTON_SCREENSHOT"] = str(args.screenshot.resolve())
    command = [str(webots), "--batch", "--mode=fast", "--stdout", "--stderr", "--minimize"]
    if not (args.rendered or args.screenshot):
        command.append("--no-rendering")
    command.append(str(world))

    scenarios.WEBOTS_CAPACITY_FILE.unlink(missing_ok=True)
    try:
        subprocess.run(command, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        print("Webots did not exit; check the controller console", file=sys.stderr)
        return 1

    capacity = scenarios.probed_webots_capacity()
    if capacity is None:
        print("probe produced no capacity file; check the Webots console for controller errors", file=sys.stderr)
        return 1
    print(scenarios.WEBOTS_CAPACITY_FILE.read_text(encoding="utf-8"))
    print(f"{scenario.name}: this host sustains {capacity} AMRs in Webots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
