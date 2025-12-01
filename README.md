# High-Energy Physics Collision Analysis Pipeline

Analysis framework for nuclear collisions using Oscar, HepMC and PHQMD formats.

## Overview

This project analyzes modified vs. unmodified particle collision events to identify and quantify cumulative scattering signatures. It processes Oscar format output files (Oscar1992A and OSC1997A), .dat for PHQMD and HepMC in parallel, extracts physics observables, and generates publication-quality comparisons.

### Key Features

- **Dual-format support**: Automatically detects and parses Oscar1992A and OSC1997A formats
- **Memory-efficient streaming**: Batch processing with configurable memory footprint
- **Collision system detection**: Automatically identifies nucleus pair and energy from particle distributions
- **Parallel processing**: Multi-worker architecture for fast throughput on large datasets
- **Physics observables**: pT broadening, rapidity distributions, cumulative signatures
- **Publication-quality plots**: Matplotlib-based visualizations with physics annotations

---

## Project Structure

```
.
├── models/
│   ├── particle.py              # Particle data class (energy-momentum, coordinates)
│   └── cumulative_signature.py   # CumulativeSignature dataclass for detected effects
│
├── utils/
│   ├── readers.py               # Oscar format readers (1992A and 1997A)
│   ├── collision_system_detector.py  # Automatic collision system identification
│   └── progress_display.py       # Live progress display for parallel workers
│
├── analyzers/
│   ├── general/
│   │   ├── aggregate_analyzer.py      # Statistical aggregation (single dataset)
│   │   └── comparison_analyzer.py     # Physics-meaningful comparison plots
│   │
│   └── cumulative/
│       └── cumulative_analyzer.py     # Cumulative effect detection & analysis
│
├── run_all_analysis.py           # Main orchestrator (parallel analysis engine)
├── verify_data_integrity.py      # Data validation tool
└── angle_analyzer.py             # Angle distribution analysis (optional)
```

---

## Core Modules

### Data Models

#### `particle.py`
Represents a single particle from collision output:
- **Kinematic**: `px`, `py`, `pz`, `E`, `mass`
- **Position**: `x`, `y`, `z`, `t_formed`
- **Properties**: `particle_id`, `charge`, `baryon_density`
- **Computed**: `pt` (transverse momentum), `eta` (pseudorapidity)

#### `cumulative_signature.py`
Detected cumulative effect signature:
- `signature_type`: Type of effect (e.g., "broadening", "deflection")
- `strength`: Magnitude 0–1
- `confidence`: Statistical confidence 0–1
- `affected_particles`: Number of particles showing effect
- `description`: Human-readable summary

### File I/O

#### `readers.py`
Handles Oscar format parsing with streaming support:

**Oscar1992A format**:
```
Event_number N_particles
particle_id mass x y z px py pz E t_formed charge baryon_density
...
```

**OSC1997A format (final_id_p_x)**:
```
final_id px py pz E mass x y z t_formed
...
```

**Key classes**:
- `OscarFormatDetector`: Auto-detects format from file header
- `Oscar1992AReader` / `OSC1997AReader`: Format-specific parsers
- `OscarReader`: Universal interface with auto-detection

**Usage**:
```python
from utils.readers import OscarReader

reader = OscarReader("path/to/file.f19")
for batch in reader.stream_batch(batch_size=500):
    for event_particles in batch:
        # Process event (list of Particle objects)
        pass
```

### Utilities

#### `collision_system_detector.py`
Infers collision parameters from event sample:

**Returns**:
- Nucleus identities (Z, A for projectile and target)
- Collision energy (√s_NN)
- Human-readable label (e.g., "Au+Au @ 10 GeV")
- Confidence score (0–1)

**Physics basis**:
- Average multiplicity → nucleus mass
- Maximum pz distribution → collision energy
- Charge distribution → nucleus identity

**Usage**:
```python
from utils.collision_system_detector import CollisionSystemDetector

system = CollisionSystemDetector.detect_from_file("run_1.f19", n_events_sample=100)
print(f"System: {system['label']} (confidence: {system['confidence']:.1%})")
```

#### `progress_display.py`
Live progress display for parallel workers using cursor control:
- Multiprocess-safe with Manager locks
- Live updates without line duplication
- Timestamp and action tracking per worker

### Analysis

#### `aggregate_analyzer.py`
Single-dataset statistical aggregation:
- Particle multiplicity distribution
- Transverse momentum (pT) spectrum: mean, std, min, max
- Rapidity (y) distribution
- Pseudorapidity (η) distribution
- Generates publication-quality histograms

**Key method**: `accumulate_event(particles)` → updates running statistics

#### `cumulative_analyzer.py`
Detects cumulative scattering signatures by comparing modified vs. unmodified:

**Physics observables**:
1. **pT broadening**: σ_pT(mod) / σ_pT(unm) — if ratio > 1.05, multiple scattering detected
2. **Mean pT shift**: Indicates energy loss vs. gain
3. **Distribution extent**: Maximum pT ratio (hard scattering signature)
4. **Angular distributions**: Forward/backward asymmetry

**Signature types**:
- `"broadening"`: Increased pT width
- `"deflection"`: Angular redistribution
- `"energy_shift"`: Mean energy change
- `"multiplicity_change"`: Particle yield modification

#### `comparison_analyzer.py`
Generates physics-meaningful comparison plots:

1. **pT Broadening (PRIMARY)**: Bar chart σ_pT(mod) vs. σ_pT(unm)
   - Detects multiple scattering
   - Threshold: 5% broadening for detection
   
2. **Mean pT Shift (SECONDARY)**: Identifies energy loss/gain
   
3. **pT Distribution Extent (TERTIARY)**: Max pT ratio
   - Shows if spectrum extends to higher pT
   
4. **Rapidity/Pseudorapidity Spectra (QUATERNARY)**: Distribution widths
   - Identifies directional effects

### Main Analysis Engine

#### `run_all_analysis.py`
Orchestrates parallel analysis pipeline:

**`UnifiedAnalysisEngine`**:
- Streams batches from modified and unmodified files
- Accumulates statistics in parallel
- Detects collision system automatically
- Generates all visualizations
- Saves results as JSON

**`_analyze_single_run(run_num)`** (multiprocessing worker):
- Processes one run (modified + unmodified file pair)
- Handles errors gracefully
- Reports progress via callback

**Workflow**:
```
For each run:
  1. Read modified file → AggregateAnalyzer (mod)
  2. Read unmodified file → AggregateAnalyzer (unm)
  3. Compare → CumulativeAnalyzer + ComparisonAnalyzer
  4. Generate plots: modified/, unmodified/, cumulative/, comparisons/
  5. Save JSON results: stats/ and cumulative/
```

---

## Usage

### Basic Example: Process Single Run

```python
from run_all_analysis import UnifiedAnalysisEngine
from pathlib import Path

data_root = Path("data")
results_root = Path("results")

engine = UnifiedAnalysisEngine(
    data_root=data_root,
    results_root=results_root,
    n_workers=8,
    batch_size=500
)

# Process runs 1–10 in parallel
engine.run_parallel(runs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

### Verify Data Integrity

Before analysis, validate that modified and unmodified files are actually different:

```bash
python verify_data_integrity.py \
  --mod data/modified/run_1.f19 \
  --unm data/no_modified/run_1.f19 \
  --events 10
```

Output:
```
Modified file: 1,000 events
Unmodified file: 1,000 events

Comparing first 10 events:
Event 0: ✓ Different (85/100 particles match)
Event 1: ✓ Different (82/100 particles match)
...

SUMMARY:
Identical events: 0
Different events: 10
✓ Files appear to be different (as expected)

STATISTICAL COMPARISON:
Modified:   pT mean = 0.4521 GeV, std = 0.3124 GeV
Unmodified: pT mean = 0.4512 GeV, std = 0.3089 GeV
ΔpT_mean: 0.000893 GeV (0.02%)
```

### Detect Collision System

```bash
python -m utils.collision_system_detector data/modified/run_1.f19
```

Output:
```
Detected Collision System:
 Label: Au+Au @ 10 GeV
 Projectile: Au (Z=79, A=197)
 Target: Au (Z=79, A=197)
 Energy: √s_NN = 10.0 GeV
 Confidence: 85%
 Events analyzed: 100
 Avg particles/event: 247
```

---

## Output Structure

After processing, results are organized as:

```
results/
├── modified/           # Modified dataset distributions
│   └── run_1/
│       ├── multiplicity.png
│       ├── pt_spectrum.png
│       ├── rapidity_distribution.png
│       └── eta_spectrum.png
│
├── unmodified/        # Unmodified dataset distributions
│   └── run_1/
│       └── [same files as modified]
│
├── cumulative/        # Cumulative effect analysis
│   ├── run_1/
│   │   ├── cumulative_effects.png
│   │   └── signatures_detected.png
│   └── run_1_cumulative.json
│
├── comparisons/       # Physics-meaningful comparisons
│   └── run_1/
│       ├── 01_pt_broadening_primary.png
│       ├── 02_mean_pt_shift.png
│       ├── 03_pt_distribution_tails.png
│       └── 04_rapidity_eta_spectra.png
│
└── stats/            # JSON summaries
    └── run_1_stats.json
```

### JSON Output Format

**`run_1_cumulative.json`**:
```json
{
  "cumulative_analysis": {
    "total_events": 1000,
    "signatures_detected": 3,
    "signature_types": ["broadening", "deflection", "energy_shift"]
  },
  "signatures": [
    {
      "type": "broadening",
      "strength": 0.87,
      "confidence": 0.92,
      "affected_particles": 2450,
      "description": "Multiple scattering detected..."
    }
  ]
}
```

---

## Physics Interpretation

### Primary Observable: pT Broadening

**Signature**: σ_pT(modified) / σ_pT(unmodified) > 1.05

**Physics**:
- Multiple scattering → transverse momentum kicks
- Cumulative effects → wider pT distribution
- Quantifies interaction strength in the medium

**Example**:
```
Modified σ_pT = 0.340 GeV
Unmodified σ_pT = 0.315 GeV
Ratio: 1.079 → ✓ Broadening detected (7.9%)
```

### Secondary Observable: Mean pT Shift

**Physics**:
- Positive shift: Energy gain (rare, fusion effects)
- Negative shift: Energy loss (ionization, radiation)
- Magnitude indicates medium density

### Tertiary Observable: Distribution Extent

**Physics**:
- Hard scattering produces high-pT particles
- Cumulative effects change spectrum endpoints
- Maximum pT ratio indicates cascade processes

---

## Performance & Memory

### Batch Processing Strategy

The pipeline uses configurable batch sizes to minimize memory:

| Batch Size | Memory per Worker | Max Events | Use Case |
|-----------|------------------|-----------|----------|
| 100 | ~200 MB | 1 million+ | Large datasets |
| 500 | ~1 GB | 100,000+ | Standard |
| 2000 | ~4 GB | 10,000 | Small, high-memory systems |

**Example for 10 GB memory limit with 8 workers**:
```python
engine = UnifiedAnalysisEngine(
    data_root=data_root,
    results_root=results_root,
    n_workers=8,
    batch_size=100  # ~200 MB × 8 = 1.6 GB comfortable headroom
)
```

### Parallel Efficiency

Typical throughput:
- **1.3 million events/hour** (8 workers, Au+Au @ 10 GeV)
- Linear scaling up to 16 workers
- I/O bound (file reading is the bottleneck)

---

## Requirements

```
Python 3.8+
numpy
matplotlib
pathlib (builtin)
```

### Installation

```bash
pip install numpy matplotlib
```

---

## Troubleshooting

### "No events found"
- **Cause**: File format not recognized or empty
- **Fix**: Run `verify_data_integrity.py` to check format
- **Check**: First 3 lines should be Oscar header

### "Modified and unmodified files are IDENTICAL"
- **Cause**: Generator not producing different outputs
- **Fix**: Verify generator configuration (energy, system, settings)
- **Check**: pT statistics should differ by >0.1%

### Memory crashes during processing
- **Cause**: Batch size too large
- **Fix**: Reduce `batch_size` parameter (default 500 → try 100)
- **Monitor**: Watch `top` or `htop` during first run

### No signatures detected
- **Cause**: Modification too weak or noise-dominated
- **Fix**: Increase sample size (more events)
- **Check**: Verify detector thresholds in `cumulative_analyzer.py`

---

## Future Extensions

- Support for arbitrary file formats (HepMC, ROOT)
- Advanced signature classification (ML-based)
- Correlation analysis across runs
- Real-time visualization server
- Batch submission to HPC clusters

---

## References

- Oscar format specifications: https://github.com/smash-transport/smash/wiki/Oscar-Format
- RHIC Beam Energy Scan: https://www.bnl.gov/bes/
- Cumulative effects in QCD: High Energy Physics literature

---

## License

Open source for research use. Please cite if used in published work.

---

## Contact & Support

For questions about physics interpretation or technical issues:
- Check `run_all_analysis.py` for detailed analysis pipeline
- Review `comparison_analyzer.py` for plot generation logic
- Consult `collision_system_detector.py` for system inference
