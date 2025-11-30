# UrQMD Cumulative Effect Analysis Framework

**Comprehensive analysis pipeline for detecting cumulative scattering signatures in heavy-ion collision simulations.**

---

## Quick Start

### Prerequisites
- Python 3.8+
- `numpy`, `matplotlib`

```bash
pip install numpy matplotlib
```

### Basic Usage

**1. Full pipeline analysis (recommended for batch processing):**
```bash
python run_all_analysis.py --data-root data --results-root results --runs 1 2 3
```

**2. Single file analysis:**
```bash
python detect_cumulative.py --modified data/modified/run_1.f19 --unmodified data/no_modified/run_1.f19 --plot all
```

**3. Verify data integrity:**
```bash
python verify_data_integrity.py --mod data/modified/run_1.f19 --unm data/no_modified/run_1.f19
```

---

## Project Overview

This framework analyzes UrQMD (Ultra-Relativistic Quantum Molecular Dynamics) collision output to identify **cumulative scattering effects**—signatures indicating multi-nucleon correlations in the collision process.

### Key Physics Concept

**Cumulative Variable:** $x = \frac{E - p_z}{m_N}$

- $x > 1$ indicates a particle carries momentum exceeding single-nucleon limits
- Requires correlation of momentum from multiple nucleons
- Forward region ($x_F > 0.3$) shows strongest signal

### Main Features

| Feature | Purpose |
|---------|---------|
| **Multi-run parallel processing** | Batch analyze 100+ runs efficiently |
| **Dual format support** | Read Oscar1992A (`.f19`) and OSC1997A formats |
| **Cumulative detection** | Identify $x > 1.1$ particles with high confidence |
| **$p_T$ broadening analysis** | Quantify momentum dispersion changes |
| **Comparison analytics** | Automatically compare modified vs. unmodified samples |
| **Cross-run aggregation** | Generate summary plots across all runs |
| **Data integrity checks** | Verify datasets differ before expensive analysis |

---

## Directory Structure

```
.
├── run_all_analysis.py           # Main entry point (parallel pipeline)
├── detect_cumulative.py          # Single-file analysis tool
├── verify_data_integrity.py      # Data validation utility
│
├── cumulative_detector.py        # Core cumulative detection logic
├── readers.py                    # Oscar format readers
├── particle.py                   # Particle model
│
├── analyzers/
│   ├── general/
│   │   ├── collision_analyzer.py     # Event-level kinematics
│   │   ├── angle_analyzer.py         # Angular distributions
│   │   ├── comparison_analyzer.py    # Modified vs. unmodified comparison
│   │   ├── aggregate_analyzer.py     # Multi-event accumulation
│   │   ├── collision_system_detector.py # Auto-detect Au+Au energy
│   │   └── cumulative_detector.py    # Cumulative signature identification
│   └── models/
│       └── particle.py
│
└── data/
    ├── modified/          # Cumulative effects enabled
    │   ├── run_1.f19
    │   ├── run_2.f19
    │   └── ...
    └── no_modified/       # Baseline (no cumulative)
        ├── run_1.f19
        ├── run_2.f19
        └── ...
```

---

## Usage Patterns

### Pattern 1: Full Pipeline (Production Use)

Process all runs in parallel with automatic system detection and cross-run comparisons:

```bash
python run_all_analysis.py \
  --data-root /path/to/data \
  --results-root /path/to/results \
  --runs 1 2 3 4 5 \
  --workers 4 \
  --batch-size 500
```

**Outputs:**
- `results/aggregate_summary.json` — Summary statistics
- `results/cross_run_aggregate_comparison.png` — Cross-run chart
- `results/modified/run_N/` — Per-run distributions
- `results/comparisons/run_N/` — Comparative plots

### Pattern 2: Single Run Diagnosis

Analyze one run pair with detailed cumulative detection:

```bash
python detect_cumulative.py \
  --modified data/modified/run_1.f19 \
  --unmodified data/no_modified/run_1.f19 \
  --plot all \
  --output cumulative_results
```

### Pattern 3: Data Validation

Before running 100+ runs, verify files are actually different:

```bash
python verify_data_integrity.py \
  --mod data/modified/run_1.f19 \
  --unm data/no_modified/run_1.f19 \
  --events 10
```

**Checks:**
- Particle multiplicity differences
- Momentum spectrum changes
- Statistical divergence

---

## Output Formats

### JSON Summary (`aggregate_summary.json`)

```json
{
  "run_1": {
    "success": true,
    "general_stats": {
      "modified": {
        "n_events": 5000,
        "n_total_particles": 150000,
        "pt": { "mean": 0.425, "std": 0.312, ... }
      }
    },
    "cumulative_analysis": {
      "likelihood": 0.82,
      "n_signatures": 3,
      "signatures": [
        {
          "type": "cumulative_variable",
          "strength": 0.95,
          "confidence": 0.95,
          "affected_particles": 2145,
          "description": "..."
        }
      ]
    }
  }
}
```

### Plot Outputs

Each run generates:
- **Distributions:** pT, rapidity ($y$), pseudorapidity ($\eta$)
- **Correlations:** 2D $y$ vs $p_T$, 2D $\eta$ vs $\phi$
- **Signatures:** Cumulative strength/confidence bar charts
- **Comparisons:** Modified vs. unmodified overlays

---

## Configuration & Customization

### Cumulative Detection Thresholds

Edit `cumulative_detector.py`:

```python
class CumulativeEffectDetector:
    def __init__(self, ...):
        self.cumulative_threshold = 1.1  # x > 1.1 for cumulative
        self.forward_cut_xf = 0.3        # x_F > 0.3 for forward
        self.cm_energy = 10.0            # √s_NN in GeV
```

### Batch Processing Parameters

```bash
--batch-size 500    # Events per memory chunk (larger = faster, uses more RAM)
--workers 4         # Parallel processes (default: CPU count)
```

### Output Directory Structure

```bash
--data-root data          # Path to data/ with modified/ and no_modified/
--results-root results    # Where to save all outputs
```

---

## Performance Tips

| Action | Effect |
|--------|--------|
| Increase `--workers` | Faster for 10+ runs (up to CPU count) |
| Increase `--batch-size` | Faster per-file processing (uses ~4 MB × batch_size) |
| Decrease `--runs` | Test on 1-2 runs before full batch |
| Use SSD for data | ~3× faster file I/O |

**Typical timing:**
- 5 runs, 5000 events each: ~30 seconds (4 workers, batch_size=500)
- 50 runs, 5000 events each: ~5 minutes (8 workers, batch_size=500)

---

## Troubleshooting

### No events found
```
Error: No events found in modified file
```
**Fix:** Ensure `.f19` files are valid Oscar1992A format. Check with:
```bash
python verify_data_integrity.py --mod file.f19 --unm baseline.f19
```

### Module import errors
```
Error importing modules: ...
```
**Fix:** Verify repository structure includes `analyzers/`, `models/`, and `utils/` directories with all `.py` files.

### Out of memory
```
Memory error with large batch
```
**Fix:** Reduce `--batch-size` or `--workers`:
```bash
python run_all_analysis.py ... --batch-size 250 --workers 2
```

---

## Citation & References

**Physics Background:**
- L. V. Bravina, E. E. Zabrodin, "Cumulative particle production in the Parton Saturation Regime"
- UrQMD: S. A. Bass et al., Prog. Part. Nucl. Phys. 41 (1998)

**Detector Algorithm:**
- Cumulative variable: $x = (E - p_L) / m_N$ (standard nuclear physics)
- Feynman scaling: $x_F = p_L / p_{L,max}$

---

## Support

For issues, questions, or contributions, please refer to the documentation directory (`docs/`) or contact the maintainer.

