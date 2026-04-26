#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path

FLOAT_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?')

def to_float(tok: str) -> float:
    return float(tok.replace('D', 'E').replace('d', 'e'))

def extract_floats(line: str):
    return [to_float(x) for x in FLOAT_RE.findall(line)]

def is_end_marker(line: str) -> bool:
    s = line.strip()
    return s.startswith('E ') or s == 'E' or s.startswith('E\t')

def is_comment_or_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    up = s.upper()
    return (
        s.startswith('#') or
        'OSCAR' in up or
        'URQMD' in up or
        'FINAL' in up
    )

def classify_particle_line(nums):
    """
    Try likely OSCAR1997A layouts.

    Official OSCAR1997A particle line in UrQMD file19:
      particle_number, particle_id, px, py, pz, E, m, x, y, z, t
    according to the manual. We also keep a few fallback layouts
    because modified generators sometimes rearrange columns.
    """
    candidates = []

    # Official-ish OSCAR97A with leading index + PDG
    if len(nums) >= 11:
        candidates.append({
            "idx": nums[0],
            "pid": nums[1],
            "px": nums[2], "py": nums[3], "pz": nums[4], "E": nums[5], "m": nums[6],
            "x": nums[7], "y": nums[8], "z": nums[9], "t": nums[10],
            "layout": "OSCAR97A idx pid px py pz E m x y z t",
        })

    # Fallback: idx pid x y z t px py pz E m
    if len(nums) >= 11:
        candidates.append({
            "idx": nums[0],
            "pid": nums[1],
            "x": nums[2], "y": nums[3], "z": nums[4], "t": nums[5],
            "px": nums[6], "py": nums[7], "pz": nums[8], "E": nums[9], "m": nums[10],
            "layout": "fallback idx pid x y z t px py pz E m",
        })

    # Fallback with no idx/pid
    if len(nums) >= 9:
        candidates.append({
            "idx": None,
            "pid": None,
            "px": nums[0], "py": nums[1], "pz": nums[2], "E": nums[3], "m": nums[4],
            "x": nums[5], "y": nums[6], "z": nums[7], "t": nums[8],
            "layout": "fallback px py pz E m x y z t",
        })
        candidates.append({
            "idx": None,
            "pid": None,
            "x": nums[0], "y": nums[1], "z": nums[2], "t": nums[3],
            "px": nums[4], "py": nums[5], "pz": nums[6], "E": nums[7], "m": nums[8],
            "layout": "fallback x y z t px py pz E m",
        })

    best = None
    best_score = -1e99

    for c in candidates:
        vals = [c["px"], c["py"], c["pz"], c["E"], c["m"], c["x"], c["y"], c["z"], c["t"]]
        if not all(math.isfinite(v) for v in vals):
            continue

        p2 = c["px"]**2 + c["py"]**2 + c["pz"]**2
        m2_calc = c["E"]**2 - p2
        rel = abs(m2_calc - c["m"]**2) / max(1.0, abs(m2_calc), abs(c["m"]**2))

        score = 0.0
        if c["E"] >= 0:
            score += 2.0
        if c["m"] >= 0:
            score += 2.0
        if c["t"] >= -1e-9:
            score += 1.0
        if m2_calc >= -1e-3:
            score += 2.0
        score += max(0.0, 6.0 - 60.0 * rel)

        if abs(c["x"]) < 1e6 and abs(c["y"]) < 1e6 and abs(c["z"]) < 1e6 and abs(c["t"]) < 1e6:
            score += 1.0

        if score > best_score:
            best_score = score
            best = c

    return best

def inspect_particle(p, args):
    issues = []

    for key in ("px", "py", "pz", "E", "m", "x", "y", "z", "t"):
        if not math.isfinite(p[key]):
            issues.append(f"non-finite {key}")

    if args.require_nonnegative_mass and p["m"] < 0:
        issues.append(f"negative mass m={p['m']:.6g}")

    if args.require_strict_positive_mass and p["m"] <= 0:
        issues.append(f"non-positive mass m={p['m']:.6g}")

    if args.require_nonnegative_time and p["t"] < 0:
        issues.append(f"negative time t={p['t']:.6g}")

    if args.require_nonnegative_energy and p["E"] < 0:
        issues.append(f"negative energy E={p['E']:.6g}")

    if args.require_nonnegative_radius0:
        for coord in ("x", "y", "z"):
            if p[coord] < 0:
                issues.append(f"negative coordinate {coord}={p[coord]:.6g}")

    p2 = p["px"]**2 + p["py"]**2 + p["pz"]**2
    m2_calc = p["E"]**2 - p2

    if m2_calc < -args.abs_m2_tol:
        issues.append(f"negative invariant m2=E^2-p^2={m2_calc:.6g}")

    if p["m"] >= 0 and math.isfinite(m2_calc):
        diff = abs(m2_calc - p["m"]**2)
        rel = diff / max(1.0, abs(m2_calc), abs(p["m"]**2))
        if rel > args.mass_tol:
            issues.append(
                f"mass-shell mismatch: m_file^2={p['m']**2:.6g}, m_calc^2={m2_calc:.6g}, rel.diff={rel:.3g}"
            )

    if p["E"] + args.abs_m2_tol < abs(p["m"]):
        issues.append(f"|m| > E : m={p['m']:.6g}, E={p['E']:.6g}")

    if p["E"] < math.sqrt(max(0.0, p2)) - args.abs_m2_tol:
        issues.append(f"E < |p| : E={p['E']:.6g}, |p|={math.sqrt(p2):.6g}")

    if abs(p["m"]) > args.huge_mass:
        issues.append(f"enormous mass |m|={abs(p['m']):.6g}")

    if abs(p["t"]) > args.huge_time:
        issues.append(f"enormous time |t|={abs(p['t']):.6g}")

    return issues

def main():
    ap = argparse.ArgumentParser(
        description="Scan UrQMD OSCAR1997A .f19 file for impossible/suspicious particles."
    )
    ap.add_argument("file", help="Path to .f19 file")
    ap.add_argument("--mass-tol", type=float, default=1e-3,
                    help="Relative tolerance for mass-shell mismatch")
    ap.add_argument("--abs-m2-tol", type=float, default=1e-6,
                    help="Absolute tolerance for negative E^2-p^2")
    ap.add_argument("--huge-mass", type=float, default=100.0,
                    help="Flag |m| above this threshold")
    ap.add_argument("--huge-time", type=float, default=1.0e6,
                    help="Flag |t| above this threshold")
    ap.add_argument("--max-report", type=int, default=200,
                    help="Maximum suspicious rows to print")

    ap.add_argument("--require-nonnegative-mass", action="store_true", default=True)
    ap.add_argument("--require-strict-positive-mass", action="store_true")
    ap.add_argument("--require-nonnegative-time", action="store_true", default=True)
    ap.add_argument("--require-nonnegative-energy", action="store_true", default=True)

    ap.add_argument("--require-nonnegative-radius0", action="store_true",
                    help="Also require x,y,z >= 0 (usually NOT recommended physically)")
    ap.add_argument("--allow-negative-mass", action="store_true",
                    help="Disable m>=0 rule")
    ap.add_argument("--allow-negative-time", action="store_true",
                    help="Disable t>=0 rule")
    ap.add_argument("--allow-negative-energy", action="store_true",
                    help="Disable E>=0 rule")

    args = ap.parse_args()

    if args.allow_negative_mass:
        args.require_nonnegative_mass = False
    if args.allow_negative_time:
        args.require_nonnegative_time = False
    if args.allow_negative_energy:
        args.require_nonnegative_energy = False

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    total_lines = 0
    particle_like = 0
    suspicious = 0
    current_event = 0
    reports = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            total_lines += 1

            if is_comment_or_header(line):
                continue
            if is_end_marker(line):
                current_event += 1
                continue

            nums = extract_floats(line)
            if len(nums) < 9:
                continue

            p = classify_particle_line(nums)
            if p is None:
                continue

            particle_like += 1
            issues = inspect_particle(p, args)
            if issues:
                suspicious += 1
                if len(reports) < args.max_report:
                    reports.append((lineno, current_event, line.strip(), p, issues))

    print(f"File: {path}")
    print(f"Total lines scanned: {total_lines}")
    print(f"Particle-like rows parsed: {particle_like}")
    print(f"Suspicious/impossible rows: {suspicious}")
    print()

    for lineno, ev, raw, p, issues in reports:
        print(f"[event {ev}, line {lineno}]")
        print(f"  issues : {'; '.join(issues)}")
        print(f"  layout : {p['layout']}")
        if p["idx"] is not None or p["pid"] is not None:
            print(f"  id     : idx={p['idx']} pid={p['pid']}")
        print(
            "  parsed : "
            f"px={p['px']:.6g} py={p['py']:.6g} pz={p['pz']:.6g} "
            f"E={p['E']:.6g} m={p['m']:.6g} "
            f"x={p['x']:.6g} y={p['y']:.6g} z={p['z']:.6g} t={p['t']:.6g}"
        )
        print(f"  raw    : {raw[:300]}")
        print()

    if suspicious > len(reports):
        print(f"... {suspicious - len(reports)} more suspicious rows not shown")

if __name__ == "__main__":
    main()