from pathlib import Path
import sys
import numpy as np
from analyzers.general.collision_analyzer import CollisionAnalyzer
from typing import List, Dict

try:
    from analyzers.general.collision_analyzer import CollisionAnalyzer
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class AggregateAnalyzer:
    """Accumulate statistics across all events in a sample."""
    
    def __init__(self, system_label: str = None, sqrt_s_NN: float = None,
                 A1: int = None, Z1: int = None, A2: int = None, Z2: int = None):
        """
        Initialize with collision system parameters (pre-detected or defaults).
        
        Args:
            system_label: Human-readable label (e.g., "Au+Au @ 10 GeV")
            sqrt_s_NN: Collision energy in GeV
            A1, Z1: Projectile mass number and charge
            A2, Z2: Target mass number and charge
        """
        self.system_label = system_label or "Unknown collision system"
        self.sqrt_s_NN = sqrt_s_NN or 10.0
        self.A1 = A1 or 197
        self.Z1 = Z1 or 79
        self.A2 = A2 or 197
        self.Z2 = Z2 or 79
        
        # Statistics storage
        self.pt_all = []
        self.y_all = []
        self.eta_all = []
        self.theta_all = []
        self.phi_all = []
        self.n_particles_per_event = []
        self.n_events = 0
        self.n_total_particles = 0
        
    def accumulate_event(self, particles: List[Particle]) -> None:
        """Add one event's particles to aggregate."""
        if not particles:
            return
            
        coll = CollisionAnalyzer(particles, self.system_label)
        
        self.pt_all.extend(coll.pt.tolist())
        self.y_all.extend(coll.y.tolist())
        self.eta_all.extend(coll.eta[np.isfinite(coll.eta)].tolist())
        self.theta_all.extend(coll.theta.tolist())
        self.phi_all.extend(coll.phi.tolist())
        self.n_particles_per_event.append(len(particles))
        self.n_events += 1
        self.n_total_particles += len(particles)
    
    def get_statistics(self) -> Dict:
        """Compute aggregate statistics."""
        if self.n_events == 0:
            return {}
        
        return {
            'n_events': self.n_events,
            'n_total_particles': self.n_total_particles,
            'system_label': self.system_label,
            'sqrt_s_NN': self.sqrt_s_NN,
            'A1': self.A1, 'Z1': self.Z1,
            'A2': self.A2, 'Z2': self.Z2,
            'n_avg_per_event': float(np.mean(self.n_particles_per_event)),
            'n_std_per_event': float(np.std(self.n_particles_per_event)),
            'pt': {
                'mean': float(np.mean(self.pt_all)),
                'std': float(np.std(self.pt_all)),
                'median': float(np.median(self.pt_all)),
                'min': float(np.min(self.pt_all)),
                'max': float(np.max(self.pt_all))
            },
            'y': {
                'mean': float(np.mean(self.y_all)),
                'std': float(np.std(self.y_all)),
                'median': float(np.median(self.y_all)),
                'min': float(np.min(self.y_all)),
                'max': float(np.max(self.y_all))
            },
            'eta': {
                'mean': float(np.mean(self.eta_all)),
                'std': float(np.std(self.eta_all)),
                'median': float(np.median(self.eta_all)),
                'min': float(np.min(self.eta_all)),
                'max': float(np.max(self.eta_all))
            },
            'theta_deg': {
                'mean': float(np.degrees(np.mean(self.theta_all))),
                'std': float(np.degrees(np.std(self.theta_all)))
            },
            'phi_deg': {
                'mean': float(np.degrees(np.mean(self.phi_all))),
                'std': float(np.degrees(np.std(self.phi_all)))
            }
        }
    
    def plot_distributions(self, output_dir: Path) -> List[str]:
        """Generate distribution plots for aggregate data."""
        import matplotlib.pyplot as plt
        
        plots = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Safety check: ensure we have data
        if not self.pt_all or not self.y_all or not self.eta_all:
            print(f"[WARNING] No data to plot: pt={len(self.pt_all)}, y={len(self.y_all)}, eta={len(self.eta_all)}")
            return plots
        
        # pT distribution - AUTO range (no fixed limits)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            pt_data = [p for p in self.pt_all if isinstance(p, (int, float)) and p > 0]
            if pt_data:
                ax.hist(pt_data, bins=150, color='steelblue', 
                        edgecolor='black', alpha=0.7, density=True)
                ax.set_xlabel(r'$p_T$ (GeV)', fontsize=12)
                ax.set_ylabel('Probability density', fontsize=12)
                ax.set_title(f'pT Distribution (all events, full range)\n{self.system_label}', fontsize=13)
                ax.set_yscale('log')
                ax.grid(alpha=0.3, axis='y')
                ax.text(0.98, 0.97, f'Events: {self.n_events}\nParticles: {self.n_total_particles}',
                        transform=ax.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            plt.tight_layout()
            pt_file = output_dir / "pt_distribution.png"
            plt.savefig(str(pt_file), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("pt_distribution.png")
        except Exception as e:
            print(f"[ERROR] pT plot failed: {e}")
        
        # Rapidity distribution - AUTO range (no fixed limits, show tails!)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            valid_y = [y for y in self.y_all if isinstance(y, (int, float))]
            if valid_y:
                ax.hist(valid_y, bins=150, color='darkgreen', 
                        edgecolor='black', alpha=0.7, density=True)
                ax.set_xlabel('Rapidity y', fontsize=12)
                ax.set_ylabel('Probability density', fontsize=12)
                ax.set_title(f'Rapidity Distribution (all events, full range)\n{self.system_label}', fontsize=13)
                ax.set_yscale('log')
                ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            y_file = output_dir / "rapidity_distribution.png"
            plt.savefig(str(y_file), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("rapidity_distribution.png")
        except Exception as e:
            print(f"[ERROR] Rapidity plot failed: {e}")
        
        # Pseudorapidity distribution - AUTO range
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            eta_data = [e for e in self.eta_all if isinstance(e, (int, float))]
            if eta_data:
                ax.hist(eta_data, bins=150, color='purple', 
                        edgecolor='black', alpha=0.7, density=True)
                ax.set_xlabel('Pseudorapidity η', fontsize=12)
                ax.set_ylabel('Probability density', fontsize=12)
                ax.set_title(f'Pseudorapidity Distribution (all events, full range)\n{self.system_label}', fontsize=13)
                ax.set_yscale('log')
                ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            eta_file = output_dir / "eta_distribution.png"
            plt.savefig(str(eta_file), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("eta_distribution.png")
        except Exception as e:
            print(f"[ERROR] Pseudorapidity plot failed: {e}")
        
        # 2D correlation: y vs pT - AUTO range
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            valid_pairs = [
                (y, p) for y, p in zip(self.y_all, self.pt_all) 
                if isinstance(y, (int, float)) and isinstance(p, (int, float)) and p > 0
            ]
            if len(valid_pairs) > 10:
                y_data, pt_data = zip(*valid_pairs)
                h = ax.hist2d(y_data, pt_data, bins=[100, 100], cmap='viridis')
                ax.set_xlabel('Rapidity y', fontsize=12)
                ax.set_ylabel(r'$p_T$ (GeV)', fontsize=12)
                ax.set_title(f'2D: y vs pT (all events, full range)\n{self.system_label}', fontsize=13)
                cbar = plt.colorbar(h[3], ax=ax)
                cbar.set_label('Counts')
            plt.tight_layout()
            y_pt_file = output_dir / "y_vs_pt.png"
            plt.savefig(str(y_pt_file), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("y_vs_pt.png")
        except Exception as e:
            print(f"[ERROR] y vs pT plot failed: {e}")
        
        # 2D correlation: η vs φ - AUTO range
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            valid_pairs = [
                (e, np.degrees(p)) for e, p in zip(self.eta_all, self.phi_all)
                if isinstance(e, (int, float)) and isinstance(p, (int, float))
            ]
            if len(valid_pairs) > 10:
                eta_data, phi_data = zip(*valid_pairs)
                h = ax.hist2d(eta_data, phi_data, bins=[100, 100], cmap='plasma')
                ax.set_xlabel('Pseudorapidity η', fontsize=12)
                ax.set_ylabel('Azimuthal angle φ (degrees)', fontsize=12)
                ax.set_title(f'2D: η vs φ (all events, full range)\n{self.system_label}', fontsize=13)
                cbar = plt.colorbar(h[3], ax=ax)
                cbar.set_label('Counts')
            plt.tight_layout()
            eta_phi_file = output_dir / "eta_vs_phi.png"
            plt.savefig(str(eta_phi_file), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("eta_vs_phi.png")
        except Exception as e:
            print(f"[ERROR] η vs φ plot failed: {e}")
        
        return plots