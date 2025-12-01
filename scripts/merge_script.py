from pathlib import Path
from typing import List


def merge_oscar_files(input_files: List[Path], output_file: Path) -> int:
    """Merge multiple OSCAR files into single file for combined analysis."""
    total_events = 0
    
    with open(output_file, 'w') as out:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                for line in infile:
                    out.write(line)
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                total_events += 1
                            except:
                                pass
    
    print(f"✓ Merged {len(input_files)} files → {output_file}")
    print(f"  Total events: {total_events}")
    return total_events

# Usage:
files_to_merge_mod = [
    Path("data/modified/run_1.f19"),
    Path("data/modified/run_2.f19"),
    Path("data/modified/run_3.f19"),
    Path("data/modified/run_4.f19"),
    Path("data/modified/run_5.f19"),
    Path("data/modified/run_6.f19"),
    Path("data/modified/run_7.f19"),
    Path("data/modified/run_8.f19"),
    Path("data/modified/run_9.f19"),
    Path("data/modified/run_10.f19"),
]
files_to_merge_unmod = [
    Path("data/no_modified/run_1.f19"),
    Path("data/no_modified/run_2.f19"),
    Path("data/no_modified/run_3.f19"),
    Path("data/no_modified/run_4.f19"),
    Path("data/no_modified/run_5.f19"),
    Path("data/no_modified/run_6.f19"),
    Path("data/no_modified/run_7.f19"),
    Path("data/no_modified/run_8.f19"),
    Path("data/no_modified/run_9.f19"),
    Path("data/no_modified/run_10.f19"),
]
merge_oscar_files(files_to_merge_mod, Path("data/modified/run_combined.f19"))
merge_oscar_files(files_to_merge_unmod, Path("data/no_modified/run_combined.f19"))