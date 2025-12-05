# API Reference

Detailed documentation of key classes and methods.

## Models

### `models.particle.Particle`

Data class representing a single particle from Oscar output.

```python
@dataclass
class Particle:
    particle_id: int              # PDG ID or custom identifier
    px: float                     # Momentum x (GeV/c)
    py: float                     # Momentum y (GeV/c)
    pz: float                     # Momentum z (GeV/c)
    mass: float                   # Invariant mass (GeV/c²)
    x: float                      # Position x (fm)
    y: float                      # Position y (fm)
    z: float                      # Position z (fm)
    E: float                      # Energy (GeV)
    t_formed: float               # Formation time (fm/c)
    charge: Optional[int]         # Electric charge
    baryon_density: Optional[float]  # Local baryon density
```

#### Properties

**`pt: float`**
- Transverse momentum: √(px² + py²)
- Signature of hard scattering and medium effects

**`eta: Optional[float]`**
- Pseudorapidity: 0.5 × ln((p + pz)/(p - pz))
- Returns None if undefined (collinear particles)
- Useful for forward/backward asymmetry studies

### `models.cumulative_signature.CumulativeSignature`

Detected cumulative scattering effect.

```python
@dataclass
class CumulativeSignature:
    signature_type: str           # Type of effect detected
    strength: float               # Magnitude of effect (0–1)
    confidence: float             # Statistical confidence (0–1)
    affected_particles: int       # Number of affected particles
    description: str              # Human-readable summary
```

#### Valid `signature_type` values

- `"broadening"`: Increased pT width (multiple scattering)
- `"deflection"`: Angular redistribution
- `"energy_shift"`: Mean energy change
- `"multiplicity_change"`: Particle yield modification

---

## Utilities

### `readers.OscarReader`

Universal Oscar format reader with automatic format detection.

```python
class OscarReader:
    def __init__(self, filepath: str, format_type: Optional[OscarFormat] = None)
    def stream_batch(self, batch_size: int) -> Generator[List[List[Particle]], None, None]
```

#### Parameters

- `filepath` (str): Path to Oscar output file
- `format_type` (OscarFormat, optional): Explicitly specify format
  - `OscarFormat.OSCAR1992A`: Standard Oscar format
  - `OscarFormat.OSC1997A`: Final state (final_id_p_x) format
  - Default: Auto-detect from file header

- `batch_size` (int): Events per batch (memory control)
  - Typical: 500 (1 GB per worker)
  - Large datasets: 100 (200 MB per worker)

#### Returns

Generator yielding batches of events:
```
batch: List[List[Particle]]
  ├─ batch[0]: Event 1 (list of Particle objects)
  ├─ batch[1]: Event 2
  └─ ...
```

#### Example

```python
from readers import OscarReader

reader = OscarReader("collision.f19")
for batch in reader.stream_batch(batch_size=500):
    for event in batch:
        # event is List[Particle]
        n_particles = len(event)
        total_momentum = sum(p.pt for p in event)
```

---

### `utils.collision_system_detector.CollisionSystemDetector`

Infers collision parameters from particle distributions.

```python
class CollisionSystemDetector:
    @staticmethod
    def detect_from_file(file_path: str, n_events_sample: int = 100) -> Dict
```

#### Parameters

- `file_path` (str): Path to Oscar file
- `n_events_sample` (int): Number of events to analyze (default: 100)
  - Typical: 100 (fast, representative)
  - Careful: 10 (very fast but unreliable)
  - Thorough: 1000 (slow but very confident)

#### Returns

Dictionary with collision parameters:

```python
{
    'A1': int,                # Projectile mass number
    'Z1': int,                # Projectile atomic number
    'A2': int,                # Target mass number
    'Z2': int,                # Target atomic number
    'sqrt_s_NN': float,       # Collision energy (GeV)
    'label': str,             # Human-readable label (e.g., "Au+Au @ 10 GeV")
    'confidence': float,      # Detection confidence (0–1)
    'n_events_analyzed': int,
    'n_particles_total': int,
    'avg_particles_per_event': float
}
```

#### Physics Inference

| Average Multiplicity | Inferred System |
|---------------------|-----------------|
| > 500 | Au+Au (gold) |
| 200–500 | Pb+Pb (lead) |
| 100–200 | Cu+Cu (copper) |
| 50–100 | p+Au (proton+gold) |
| < 50 | Default: Au+Au |

#### Energy Estimation

| Max pz (GeV) | Inferred √s_NN |
|-------------|----------------|
| < 2 | 5 GeV |
| 2–5 | 10 GeV |
| 5–10 | 20 GeV |
| 10–50 | 50 GeV |
| 50–100 | 200 GeV |
| > 100 | 2760 GeV (LHC) |

#### Example

```python
from utils.collision_system_detector import CollisionSystemDetector

system = CollisionSystemDetector.detect_from_file(
    "run_1.f19",
    n_events_sample=100
)

print(f"System: {system['label']}")
print(f"Projectile: {system['nucleus_name'](system['Z1'], system['A1'])}")
print(f"Confidence: {system['confidence']:.1%}")

if system['confidence'] < 0.5:
    print("Warning: Low confidence, results should be interpreted carefully")
```

---

## Analyzers

### `analyzers.general.aggregate_analyzer.AggregateAnalyzer`

Accumulates statistics from a single dataset.

```python
class AggregateAnalyzer:
    def __init__(self, system_label: str = None, sqrt_s_NN: float = None,
                 A1: int = None, Z1: int = None, A2: int = None, Z2: int = None)
    
    def accumulate_event(self, particles: List[Particle]) -> None
    def get_statistics(self) -> Dict
    def plot_distributions(self, output_dir: Path) -> None
```

#### Methods

**`accumulate_event(particles: List[Particle])`**
- Updates running statistics with one event
- Call once per event (inside your processing loop)

**`get_statistics() -> Dict`**
Returns aggregate statistics:
```python
{
    'multiplicity': {
        'mean': float,
        'std': float,
        'min': int,
        'max': int
    },
    'pt': {
        'mean': float,     # Average transverse momentum
        'std': float,      # Width of pT distribution
        'min': float,
        'max': float
    },
    'y': {                 # Rapidity
        'mean': float,
        'std': float,
        ...
    },
    'eta': {               # Pseudorapidity
        'mean': float,
        'std': float,
        ...
    },
    'system_label': str,
    'sqrt_s_NN': float
}
```

**`plot_distributions(output_dir: Path)`**
- Generates histogram plots:
  - `multiplicity.png`: Event-by-event particle count
  - `pt_spectrum.png`: pT distribution
  - `rapidity_distribution.png`: Rapidity spectrum
  - `eta_spectrum.png`: Pseudorapidity spectrum

#### Example

```python
from analyzers.general.aggregate_analyzer import AggregateAnalyzer
from readers import OscarReader

analyzer = AggregateAnalyzer(system_label="Au+Au", sqrt_s_NN=10.0)
reader = OscarReader("events.f19")

for batch in reader.stream_batch(batch_size=500):
    for event in batch:
        analyzer.accumulate_event(event)

stats = analyzer.get_statistics()
print(f"Mean pT: {stats['pt']['mean']:.4f} GeV")
print(f"pT width (σ): {stats['pt']['std']:.4f} GeV")

analyzer.plot_distributions("output/")
```

---

### `analyzers.general.comparison_analyzer.RunComparisonAnalyzer`

Generates physics-meaningful comparison plots.

```python
class RunComparisonAnalyzer:
    def __init__(self, mod_stats: Dict, unm_stats: Dict, run_name: str)
    
    def plot_pt_broadening_ratio(self, output_file: Optional[str] = None) -> None
    def plot_mean_pt_shift(self, output_file: Optional[str] = None) -> None
    def plot_distribution_tails(self, output_file: Optional[str] = None) -> None
    def plot_rapidity_eta_spectra(self, output_file: Optional[str] = None) -> None
    def generate_all_comparisons(self, output_dir: Path) -> List[str]
```

#### Physics Plots

**`plot_pt_broadening_ratio()` — PRIMARY OBSERVABLE**
- Compares pT width: σ(mod) vs σ(unm)
- Threshold: Ratio > 1.05 indicates multiple scattering
- Physics: Cumulative interactions increase transverse momentum kicks

**`plot_mean_pt_shift()` — SECONDARY**
- Mean pT difference between datasets
- Detects energy loss (negative shift) or gain (positive shift)

**`plot_distribution_tails()` — TERTIARY**
- Maximum pT ratio (mod/unm)
- Identifies hard scattering signature and spectrum extent

**`plot_rapidity_eta_spectra()` — QUATERNARY**
- Rapidity and pseudorapidity distributions
- Identifies forward/backward asymmetry

#### Example

```python
from analyzers.general.comparison_analyzer import RunComparisonAnalyzer

stats_modified = {...}    # From AggregateAnalyzer
stats_unmodified = {...}

comparator = RunComparisonAnalyzer(
    mod_stats=stats_modified,
    unm_stats=stats_unmodified,
    run_name="run_1"
)

# Generate all comparison plots
plots = comparator.generate_all_comparisons("output/comparisons/")

for plot in plots:
    print(f"Generated: {plot}")
```

---

### `analyzers.cumulative.cumulative_analyzer.CumulativeAnalyzer`

Detects cumulative scattering effects by comparing batches.

```python
class CumulativeAnalyzer:
    def process_batch(self, batch_mod: List[List[Particle]],
                      batch_unm: List[List[Particle]]) -> None
    
    def get_signatures(self) -> List[CumulativeSignature]
    def get_statistics(self) -> Dict
    def plot_distributions(self, output_dir: Path) -> None
```

#### Methods

**`process_batch(batch_mod, batch_unm)`**
- Analyzes one batch of modified and unmodified events
- Detects signatures incrementally

**`get_signatures() -> List[CumulativeSignature]`**
- Returns all detected cumulative signatures
- Each signature includes type, strength, confidence, affected particle count

**`get_statistics() -> Dict`**
- Summary statistics and signature counts

#### Example

```python
from analyzers.cumulative.cumulative_analyzer import CumulativeAnalyzer

analyzer = CumulativeAnalyzer()

reader_mod = OscarReader("modified.f19")
reader_unm = OscarReader("unmodified.f19")

for batch_mod, batch_unm in zip(
    reader_mod.stream_batch(500),
    reader_unm.stream_batch(500)
):
    analyzer.process_batch(batch_mod, batch_unm)

signatures = analyzer.get_signatures()
for sig in signatures:
    print(f"{sig.signature_type}: strength={sig.strength:.2f}, "
          f"confidence={sig.confidence:.2f}")
```

---

## Main Engine

### `run_all_analysis.UnifiedAnalysisEngine`

Orchestrates full analysis pipeline.

```python
class UnifiedAnalysisEngine:
    def __init__(self, data_root: Path, results_root: Path,
                 n_workers: Optional[int] = None, batch_size: int = 500)
    
    def run_parallel(self, runs: List[int]) -> None
```

#### Parameters

- `data_root` (Path): Root directory containing data/
  - Expected structure:
    ```
    data_root/
    ├── modified/
    │   ├── run_1.f19
    │   ├── run_2.f19
    │   └── ...
    └── no_modified/
        ├── run_1.f19
        ├── run_2.f19
        └── ...
    ```

- `results_root` (Path): Output directory for results

- `n_workers` (int, optional): Number of parallel workers
  - Default: CPU count
  - Recommended: 8 for 16-core system

- `batch_size` (int): Events per batch
  - Default: 500
  - Adjust for memory constraints

#### Methods

**`run_parallel(runs: List[int])`**
- Main entry point
- Processes multiple runs in parallel
- Automatically detects collision system
- Generates all plots and JSON results

#### Example

```python
from run_all_analysis import UnifiedAnalysisEngine
from pathlib import Path

engine = UnifiedAnalysisEngine(
    data_root=Path("data"),
    results_root=Path("results"),
    n_workers=8,
    batch_size=500
)

# Process runs 1–100
engine.run_parallel(runs=list(range(1, 101)))
```

---

## Utilities

### `utils.progress_display.ProgressDisplay`

Live progress display for parallel processing.

```python
class ProgressDisplay:
    @staticmethod
    def initialize(num_runs: int) -> Tuple[Lock, Dict, int]
    
    @staticmethod
    def report(run_num: int, action: str, progress: str,
               lock, states, total) -> None
```

#### Example (For Custom Scripts)

```python
from utils.progress_display import ProgressDisplay

lock, states, total = ProgressDisplay.initialize(10)

# In worker thread/process:
ProgressDisplay.report(
    run_num=1,
    action="Processing",
    progress="500/1000 events",
    lock=lock,
    states=states,
    total=total
)
```

Output:
```
[12:34:56] [run_1] Processing        | 500/1000 events
[12:34:56] [run_2] Reading           | ...
...
```

---

## Data Validation

### `verify_data_integrity.compare_file_pair()`

Validates that modified and unmodified files are actually different.

```python
def compare_file_pair(mod_file: str, unm_file: str, n_events: int = 5) -> None
```

#### Command-line Usage

```bash
python verify_data_integrity.py \
  --mod data/modified/run_1.f19 \
  --unm data/no_modified/run_1.f19 \
  --events 10
```

#### Output Analysis

- **Identical events**: Should be 0 (otherwise files are the same)
- **Different events**: Should be > 0 (modification is working)
- **pT statistics**: ΔpT_mean > 0.1% is typical for real modifications

---

## JSON Output Format

### `run_N_stats.json`

```json
{
  "run_name": "run_1",
  "modified": {
    "multiplicity": {"mean": 247.3, "std": 45.2, "min": 150, "max": 520},
    "pt": {"mean": 0.4521, "std": 0.3124, "min": 0.001, "max": 8.234},
    "y": {"mean": 0.023, "std": 1.854, "min": -5.2, "max": 5.1},
    "eta": {"mean": 0.045, "std": 1.876, "min": -5.4, "max": 5.3},
    "system_label": "Au+Au @ 10 GeV",
    "sqrt_s_NN": 10.0
  },
  "unmodified": { ... },
  "cumulative": { ... }
}
```

### `run_N_cumulative.json`

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
      "description": "Multiple scattering detected: σ_pT(mod)/σ_pT(unm) = 1.078"
    },
    { ... }
  ]
}
```

---

## Error Handling

All analyzers include error handling for:

- Empty events (no particles)
- Invalid particle properties (NaN, division by zero)
- File I/O errors (missing files, permission denied)
- Parsing errors (malformed lines, unexpected format)

Graceful degradation:
- Skips problematic events
- Logs warnings
- Continues with valid data
- Reports final status
