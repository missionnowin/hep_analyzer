from pathlib import Path
import sys
import numpy as np
from typing import List, Dict, Optional

try:
    from analyzers.general.collision_analyzer import CollisionAnalyzer
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


from pathlib import Path
import sys
import numpy as np
from typing import List, Dict

try:
    from analyzers.general.collision_analyzer import CollisionAnalyzer
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class AggregateAnalyzer:
    """Accumulate statistics across all events - OPTIMIZED with NumPy arrays.
    """

    def __init__(
        self,
        system_label: str = None,
        sqrt_s_NN: float = None,
        A1: int = None,
        Z1: int = None,
        A2: int = None,
        Z2: int = None,
        initial_capacity: int = 1_000_000,
    ):
        """
        Initialize with collision system parameters.

        Args:
            system_label: Human-readable label (e.g., "Au+Au @ 10 GeV")
            sqrt_s_NN: Collision energy in GeV
            A1, Z1: Projectile mass number and charge
            A2, Z2: Target mass number and charge
            initial_capacity: Pre-allocate space for this many particles (grows if exceeded)
        """
        self.system_label = system_label or "Unknown collision system"
        self.sqrt_s_NN = sqrt_s_NN or 10.0
        self.A1 = A1 or 197
        self.Z1 = Z1 or 79
        self.A2 = A2 or 197
        self.Z2 = Z2 or 79

        # Pre-allocate NumPy arrays with float32 (50% more compact than float64)
        self._capacity = initial_capacity
        self.pt_all = np.empty(self._capacity, dtype=np.float32)
        self.y_all = np.empty(self._capacity, dtype=np.float32)
        self.eta_all = np.empty(self._capacity, dtype=np.float32)
        self.theta_all = np.empty(self._capacity, dtype=np.float32)
        self.phi_all = np.empty(self._capacity, dtype=np.float32)

        # Track current fill level
        self._count = 0
        self.n_particles_per_event = []
        self.n_events = 0
        self.n_total_particles = 0

        self.particles_modified = []     
        self.particles_unmodified = []  

    def _ensure_capacity(self, required_size: int) -> None:
        """Grow arrays if capacity exceeded."""
        if required_size > self._capacity:
            new_capacity = int(self._capacity * 1.5)
            self.pt_all = np.resize(self.pt_all, new_capacity)
            self.y_all = np.resize(self.y_all, new_capacity)
            self.eta_all = np.resize(self.eta_all, new_capacity)
            self.theta_all = np.resize(self.theta_all, new_capacity)
            self.phi_all = np.resize(self.phi_all, new_capacity)
            self._capacity = new_capacity
    

    def accumulate_event(self, particles: List[Particle], isModified: bool) -> None:
        """Add one event's particles to aggregate using in-place array operations."""
        if not particles:
            return

        coll = CollisionAnalyzer(particles, self.system_label)

        if isModified:
            self.particles_modified.extend(particles) 
        else:
            self.particles_unmodified.extend(particles) 

        # Get values and ensure float32 dtype
        pt_vals = coll.pt.astype(np.float32)
        y_vals = coll.y.astype(np.float32)
        eta_vals = coll.eta[np.isfinite(coll.eta)].astype(np.float32)
        theta_vals = coll.theta.astype(np.float32)
        phi_vals = coll.phi.astype(np.float32)

        # Ensure capacity for all particles
        new_count = self._count + len(pt_vals)
        self._ensure_capacity(new_count)

        # In-place slice assignment (no copying, no reallocation)
        self.pt_all[self._count : new_count] = pt_vals
        self.y_all[self._count : new_count] = y_vals
        self.theta_all[self._count : new_count] = theta_vals
        self.phi_all[self._count : new_count] = phi_vals

        # Handle eta: variable length due to NaN filtering
        n_eta = len(eta_vals)
        if n_eta > 0:
            self.eta_all[self._count : self._count + n_eta] = eta_vals
            if n_eta < len(pt_vals):
                self.eta_all[self._count + n_eta : new_count] = np.nan

        # Update counters
        self._count = new_count
        self.n_particles_per_event.append(len(particles))
        self.n_events += 1
        self.n_total_particles += len(particles)

        # Optional: Clear analyzer cache to free memory
        if hasattr(coll, 'clear_cache'):
            coll.clear_cache()


    def get_statistics(self) -> Dict:
        """Compute aggregate statistics from accumulated data."""
        if self.n_events == 0:
            return {}

        # Trim arrays to actual size for statistics
        pt_data = self.pt_all[: self._count]
        y_data = self.y_all[: self._count]
        eta_data = self.eta_all[: self._count]
        theta_data = self.theta_all[: self._count]
        phi_data = self.phi_all[: self._count]

        # Filter valid values
        valid_eta = eta_data[np.isfinite(eta_data)]

        return {
            "n_events": self.n_events,
            "n_total_particles": self.n_total_particles,
            "system_label": self.system_label,
            "sqrt_s_NN": self.sqrt_s_NN,
            "A1": self.A1,
            "Z1": self.Z1,
            "A2": self.A2,
            "Z2": self.Z2,
            "n_avg_per_event": float(np.mean(self.n_particles_per_event)),
            "n_std_per_event": float(np.std(self.n_particles_per_event)),
            "pt": {
                "mean": float(np.mean(pt_data)),
                "std": float(np.std(pt_data)),
                "median": float(np.median(pt_data)),
                "min": float(np.min(pt_data)),
                "max": float(np.max(pt_data)),
            },
            "y": {
                "mean": float(np.mean(y_data)),
                "std": float(np.std(y_data)),
                "median": float(np.median(y_data)),
                "min": float(np.min(y_data)),
                "max": float(np.max(y_data)),
            },
            "eta": {
                "mean": float(np.mean(valid_eta)) if len(valid_eta) > 0 else 0.0,
                "std": float(np.std(valid_eta)) if len(valid_eta) > 0 else 0.0,
                "median": float(np.median(valid_eta)) if len(valid_eta) > 0 else 0.0,
                "min": float(np.min(valid_eta)) if len(valid_eta) > 0 else 0.0,
                "max": float(np.max(valid_eta)) if len(valid_eta) > 0 else 0.0,
            },
            "theta_deg": {
                "mean": float(np.degrees(np.mean(theta_data))),
                "std": float(np.degrees(np.std(theta_data))),
            },
            "phi_deg": {
                "mean": float(np.degrees(np.mean(phi_data))),
                "std": float(np.degrees(np.std(phi_data))),
            },
        }
    

    def plot_distributions(self, output_dir: Path) -> List[str]:
        """Generate distribution plots for aggregate data."""
        import matplotlib.pyplot as plt

        plots = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # Trim to actual data
        pt_data = self.pt_all[: self._count]
        y_data = self.y_all[: self._count]
        eta_data = self.eta_all[: self._count]
        theta_data = self.theta_all[: self._count]
        phi_data = self.phi_all[: self._count]

        if len(pt_data) == 0:
            print(f"[WARNING] No data to plot")
            return plots

        # pT distribution
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            pt_valid = pt_data[pt_data > 0]
            if len(pt_valid) > 0:
                ax.hist(
                    pt_valid,
                    bins=150,
                    color="steelblue",
                    edgecolor="black",
                    alpha=0.7,
                    density=True,
                )
                ax.set_xlabel(r"$p_T$ (GeV)", fontsize=12)
                ax.set_ylabel("Probability density", fontsize=12)
                ax.set_title(f"pT Distribution (all events)\n{self.system_label}", fontsize=13)
                ax.set_yscale("log")
                ax.grid(alpha=0.3, axis="y")
                ax.text(
                    0.98,
                    0.97,
                    f"Events: {self.n_events}\nParticles: {self.n_total_particles}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )
                plt.tight_layout()
                pt_file = output_dir / "pt_distribution.png"
                plt.savefig(str(pt_file), dpi=300, bbox_inches="tight")
                plt.close()
                plots.append("pt_distribution.png")
        except Exception as e:
            print(f"[ERROR] pT plot failed: {e}")

        # Rapidity distribution
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            y_valid = y_data[np.isfinite(y_data)]
            if len(y_valid) > 0:
                ax.hist(y_valid, bins=150, color="darkgreen", edgecolor="black", alpha=0.7, density=True)
                ax.set_xlabel("Rapidity y", fontsize=12)
                ax.set_ylabel("Probability density", fontsize=12)
                ax.set_title(f"Rapidity Distribution (all events)\n{self.system_label}", fontsize=13)
                ax.set_yscale("log")
                ax.grid(alpha=0.3, axis="y")
                plt.tight_layout()
                y_file = output_dir / "rapidity_distribution.png"
                plt.savefig(str(y_file), dpi=300, bbox_inches="tight")
                plt.close()
                plots.append("rapidity_distribution.png")
        except Exception as e:
            print(f"[ERROR] Rapidity plot failed: {e}")

        # Pseudorapidity distribution
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            eta_valid = eta_data[np.isfinite(eta_data)]
            if len(eta_valid) > 0:
                ax.hist(eta_valid, bins=150, color="purple", edgecolor="black", alpha=0.7, density=True)
                ax.set_xlabel("Pseudorapidity η", fontsize=12)
                ax.set_ylabel("Probability density", fontsize=12)
                ax.set_title(f"Pseudorapidity Distribution (all events)\n{self.system_label}", fontsize=13)
                ax.set_yscale("log")
                ax.grid(alpha=0.3, axis="y")
                plt.tight_layout()
                eta_file = output_dir / "eta_distribution.png"
                plt.savefig(str(eta_file), dpi=300, bbox_inches="tight")
                plt.close()
                plots.append("eta_distribution.png")
        except Exception as e:
            print(f"[ERROR] Pseudorapidity plot failed: {e}")

        # 2D: y vs pT
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            valid_pairs = [
                (y, p)
                for y, p in zip(y_data, pt_data)
                if np.isfinite(y) and np.isfinite(p) and p > 0
            ]
            if len(valid_pairs) > 10:
                y_vals, pt_vals = zip(*valid_pairs)
                h = ax.hist2d(y_vals, pt_vals, bins=[100, 100], cmap="viridis")
                ax.set_xlabel("Rapidity y", fontsize=12)
                ax.set_ylabel(r"$p_T$ (GeV)", fontsize=12)
                ax.set_title(f"2D: y vs pT (all events)\n{self.system_label}", fontsize=13)
                cbar = plt.colorbar(h[3], ax=ax)
                cbar.set_label("Counts")
                plt.tight_layout()
                y_pt_file = output_dir / "y_vs_pt.png"
                plt.savefig(str(y_pt_file), dpi=300, bbox_inches="tight")
                plt.close()
                plots.append("y_vs_pt.png")
        except Exception as e:
            print(f"[ERROR] y vs pT plot failed: {e}")

        # 2D: η vs φ
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            valid_pairs = [
                (e, np.degrees(p))
                for e, p in zip(eta_data, phi_data)
                if np.isfinite(e) and np.isfinite(p)
            ]
            if len(valid_pairs) > 10:
                eta_vals, phi_vals = zip(*valid_pairs)
                h = ax.hist2d(eta_vals, phi_vals, bins=[100, 100], cmap="plasma")
                ax.set_xlabel("Pseudorapidity η", fontsize=12)
                ax.set_ylabel("Azimuthal angle φ (degrees)", fontsize=12)
                ax.set_title(f"2D: η vs φ (all events)\n{self.system_label}", fontsize=13)
                cbar = plt.colorbar(h[3], ax=ax)
                cbar.set_label("Counts")
                plt.tight_layout()
                eta_phi_file = output_dir / "eta_vs_phi.png"
                plt.savefig(str(eta_phi_file), dpi=300, bbox_inches="tight")
                plt.close()
                plots.append("eta_vs_phi.png")
        except Exception as e:
            print(f"[ERROR] η vs φ plot failed: {e}")

        return plots


    def detect_cumulative_effects(
        self,
        particles_unmodified: Optional[List[Particle]] = None
    ) -> Dict:
        """Detect cumulative effects by comparing modified vs unmodified particles
        
        Should be called AFTER accumulate_event() for all events.
        Uses particles already accumulated in self.particles_modified.
        """
        if not self.particles_modified:
            return {
                "likelihood": 0.0,
                "n_signatures": 0,
                "signatures": [],
                "error": "No particles to analyze"
            }
        
        try:
            from analyzers.cumulative.cumulative_detector import CumulativeEffectDetector
        except ImportError as e:
            return {
                "likelihood": 0.0,
                "n_signatures": 0,
                "signatures": [],
                "error": f"Cannot import CumulativeEffectDetector: {e}"
            }
        
        detector = CumulativeEffectDetector(
            particles_modified=self.particles_modified,
            particles_unmodified=particles_unmodified or []
        )
        
        detector.detect_all_signatures()
        cum_likelihood = float(detector.get_cumulative_likelihood())
        
        # Extract data BEFORE deleting detector
        n_signatures = len(detector.signatures)
        n_particles_mod = len(self.particles_modified)
        n_particles_unm = len(particles_unmodified) if particles_unmodified else 0
        
        signatures_list = [
            {
                "type": sig.signature_type,
                "strength": float(sig.strength),
                "confidence": float(sig.confidence),
                "affected_particles": int(sig.affected_particles),
                "description": sig.description,
            }
            for sig in detector.signatures
        ]
        
        # Safe to delete now
        del detector
        import gc
        gc.collect()
        
        # Build result using extracted variables
        result = {
            "likelihood": cum_likelihood,
            "n_signatures": n_signatures,
            "n_particles_mod": n_particles_mod,
            "n_particles_unm": n_particles_unm,
            "signatures": signatures_list,
        }
        
        return result


    def plot_cumulative_summary(
        self,
        output_dir,
        particles_unmodified: Optional[List[Particle]] = None
    ) -> List[str]:
        """Generate cumulative analysis plots
        
        Args:
            output_dir: Directory to save plots
            particles_unmodified: Optional unmodified particles for comparison
            
        Returns:
            List of generated plot filenames
        """
        plots = []
        
        if not self.particles_modified:
            return plots
        
        try:
            from analyzers.cumulative.cumulative_detector import CumulativeEffectDetector
        except ImportError:
            return plots
        
        detector = CumulativeEffectDetector(
            particles_modified=self.particles_modified or [],
            particles_unmodified=particles_unmodified or []
        )
        detector.detect_all_signatures()
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_file = output_dir / "cumulative_summary.png"
            detector.plot_cumulative_summary(out=str(summary_file))
            plots.append("cumulative_summary.png")
        except Exception as e:
            print(f"[ERROR] Cumulative summary plot failed: {e}")
        
        try:
            candidates_file = output_dir / "cumulative_candidates.png"
            detector.plot_cumulative_candidates(out=str(candidates_file))
            plots.append("cumulative_candidates.png")
        except Exception as e:
            print(f"[ERROR] Cumulative candidates plot failed: {e}")
        
        # Safe cleanup after all plots done
        del detector
        import gc
        gc.collect()
        
        return plots


    def get_cumulative_statistics(
        self,
        particles_unmodified: Optional[List[Particle]] = None
    ) -> Dict:
        """Get comprehensive cumulative effect statistics
        
        Returns dict with all cumulative analysis metrics
        """
        return self.detect_cumulative_effects(particles_unmodified)


    def clear_cache(self):
        """Explicitly clear cache to free memory."""
        pass  # NumPy arrays are freed automatically

    def __del__(self):
        """Cleanup on deletion."""
        pass  # NumPy arrays freed automatically by garbage collector