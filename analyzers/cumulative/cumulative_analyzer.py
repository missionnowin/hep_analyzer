import sys
from typing import Dict, List, Tuple
from pathlib import Path
from scipy.ndimage import gaussian_filter

from matplotlib import cm
from matplotlib.patches import Patch
import numpy as np

from models.ions import CHARGED_PDGIDS, PROTONS

try:
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CumulativeAnalyzer:
    def __init__(self):
        self.total_events_mod = 0
        self.total_events_unm = 0

        self.total_particles_mod = 0
        self.total_particles_unm = 0

        self.multiplicity_mod = []
        self.multiplicity_unm = []
        
        # Angular distributions
        self.theta_dist_mod = []
        self.theta_dist_unm = []

        self.theta_dist_charged_mod = []
        self.theta_dist_charged_unmod = []

        self.theta_dist_protons_mod = []
        self.theta_dist_protons_unmod = []

        # Kinematic phase space (kz vs k_perp) scatter data
        self.kz_kperp_charged_mod: List[Tuple[float, float]] = []
        self.kz_kperp_charged_unmod: List[Tuple[float, float]] = []

        self.kz_kperp_protons_mod: List[Tuple[float, float]] = []
        self.kz_kperp_protons_unmod: List[Tuple[float, float]] = []

        self._finalized = False
    

    def process_batch(self, batch_mod: List, batch_unm: List) -> None:
        """
        Process a batch of events from modified and unmodified data.
        """
        if not batch_mod or not batch_unm:
            return
        
        for particles_mod, particles_unm in zip(batch_mod, batch_unm):
            self.total_events_mod += 1
            self.total_events_unm += 1
            
            n_mod = len(particles_mod) if particles_mod else 0
            n_unm = len(particles_unm) if particles_unm else 0
            
            self.total_particles_mod += n_mod
            self.total_particles_unm += n_unm
            
            self.multiplicity_mod.append(n_mod)
            self.multiplicity_unm.append(n_unm)
        
        # Accumulate data only
        self._accumulate_from_batch(batch_mod, batch_unm)
    

    def _accumulate_from_batch(self, batch_mod: List[List[Particle]], batch_unm: List[List[Particle]]) -> None:
        """Accumulate forbidden particles and angular samples from batch."""
        # Accumulate data
        for event in batch_mod:
            if event:
                for p in event:
                    theta, _ = self._get_angles_from_particle(p)
                    if theta is not None:
                        self.theta_dist_mod.append(theta)
                        kz = p.pz
                        kperp = (p.px**2 + p.py**2) ** 0.5
                        
                        if p.particle_id in CHARGED_PDGIDS and p.E > 0.3 and kz < 0:
                            self.theta_dist_charged_mod.append(theta)

                            kz = p.pz
                            kperp = (p.px**2 + p.py**2) ** 0.5

                            self.kz_kperp_charged_mod.append((kz, kperp))
                            if p.particle_id in PROTONS:
                                self.theta_dist_protons_mod.append(theta)
                                self.kz_kperp_protons_mod.append((kz, kperp))
        
        for event in batch_unm:
            if event:
                for p in event:
                    theta, _ = self._get_angles_from_particle(p)
                    if theta is not None:
                        self.theta_dist_unm.append(theta)
                        kz = p.pz
                        kperp = (p.px**2 + p.py**2) ** 0.5
                        
                        if p.particle_id in CHARGED_PDGIDS and p.E > 0.3 and kz < 0:
                            self.theta_dist_charged_unmod.append(theta)

                            self.kz_kperp_charged_unmod.append((kz, kperp))
                            if p.particle_id in PROTONS:
                                self.theta_dist_protons_unmod.append(theta)
                                self.kz_kperp_protons_unmod.append((kz, kperp))

    
    def _get_angles_from_particle(self, p: Particle) -> Tuple:
        """
        Calculate theta (polar angle) and phi (azimuthal angle) from particle momentum.
        Returns (theta, phi) in radians or (None, None) if invalid.
        """
        p_total = (p.px**2 + p.py**2 + p.pz**2) ** 0.5
        
        if p_total == 0:
            return None, None
        
        theta = np.arccos(np.clip(p.pz / p_total, -1.0, 1.0))
        phi = np.arctan2(p.py, p.px)
        
        return theta, phi
    
    def plot_distributions(self, output_dir) -> List[Path]:
        """
        Generate plots comparing modified vs unmodified distributions.
        Returns list of plot file paths.
        """
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
        except ImportError:
            print("Matplotlib not available - skipping plots")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = []
        
        # Plot 1: Angular Distribution (Theta)
        if len(self.theta_dist_mod) > 0 and len(self.theta_dist_unm) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bins = np.linspace(0, np.pi, 50)
            ax.hist(self.theta_dist_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.theta_dist_unm, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            
            ax.set_xlabel('Scattering Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "01_theta_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)

        # Plot 2_1: Angular Distribution (Theta)
        if len(self.theta_dist_charged_mod) > 0 and len(self.theta_dist_charged_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bins = np.linspace(0, np.pi, 50)
            ax.hist(self.theta_dist_charged_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.theta_dist_charged_unmod, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            
            ax.set_xlabel('Scattering Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison For Charged Particles', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "02_a_theta_distribution_charged.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)

        # Plot 2_3: Angular Distribution (Theta)
        if len(self.theta_dist_protons_mod) > 0 and len(self.theta_dist_protons_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bins = np.linspace(0, np.pi, 50)
            ax.hist(self.theta_dist_protons_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.theta_dist_protons_unmod, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            
            ax.set_xlabel('Scattering Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison For Protons', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "02_c_theta_distribution_protons.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
# Plot 2_5: kz vs k_perp for charged particles
        if len(self.kz_kperp_charged_mod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))


            kz_mod = [point[0] for point in self.kz_kperp_charged_mod]
            kperp_mod = [point[1] for point in self.kz_kperp_charged_mod]


            ax.scatter(kz_mod, kperp_mod, s=8, alpha=0.5, label='Modified', color='red')


            ax.set_xlabel('k_z', fontsize=12)
            ax.set_ylabel('k_perpendicular', fontsize=12)
            ax.set_title('k_z vs k_perpendicular For Charged Particles', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)


            plot_path = output_dir / "02_e_kz_kperp_charged_mod.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
       
        # Plot 2_5: kz vs k_perp for charged particles
        if len(self.kz_kperp_charged_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))


            kz_unm = [point[0] for point in self.kz_kperp_charged_unmod]
            kperp_unm = [point[1] for point in self.kz_kperp_charged_unmod]


            ax.scatter(kz_unm, kperp_unm, s=8, alpha=0.5, label='Unmodified', color='blue')


            ax.set_xlabel('k_z', fontsize=12)
            ax.set_ylabel('k_perpendicular', fontsize=12)
            ax.set_title('k_z vs k_perpendicular For Charged Particles', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)


            plot_path = output_dir / "02_e_kz_kperp_charged_unmod.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)


        # Plot 2_5: kz vs k_perp for charged particles
        if len(self.kz_kperp_charged_mod) > 0 and len(self.kz_kperp_charged_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))


            kz_unm = [point[0] for point in self.kz_kperp_charged_unmod]
            kz_mod = [point[0] for point in self.kz_kperp_charged_mod]


            kperp_unm = [point[1] for point in self.kz_kperp_charged_unmod]
            kperp_mod = [point[1] for point in self.kz_kperp_charged_mod]


            ax.scatter(kz_mod, kperp_mod, s=8, alpha=0.5, label='Modified', color='red')
            ax.scatter(kz_unm, kperp_unm, s=8, alpha=0.5, label='Unmodified', color='blue')


            ax.set_xlabel('k_z', fontsize=12)
            ax.set_ylabel('k_perpendicular', fontsize=12)
            ax.set_title('k_z vs k_perpendicular For Charged Particles', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)


            plot_path = output_dir / "02_e_kz_kperp_charged_mod_unmod.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)


        # Plot 2_7: kz vs k_perp for protons
        if len(self.kz_kperp_protons_mod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))


            kz_mod = [point[0] for point in self.kz_kperp_protons_mod]
            kperp_mod = [point[1] for point in self.kz_kperp_protons_mod]


            ax.scatter(kz_mod, kperp_mod, s=8, alpha=0.5, label='Modified', color='red')


            ax.set_xlabel('k_z', fontsize=12)
            ax.set_ylabel('k_perpendicular', fontsize=12)
            ax.set_title('k_z vs k_perpendicular For Protons', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)


            plot_path = output_dir / "02_g_kz_kperp_protons_mod.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
       
        if len(self.kz_kperp_protons_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))


            kz_unm = [point[0] for point in self.kz_kperp_protons_unmod]
            kperp_unm = [point[1] for point in self.kz_kperp_protons_unmod]


            ax.scatter(kz_unm, kperp_unm, s=8, alpha=0.5, label='Unmodified', color='blue')


            ax.set_xlabel('k_z', fontsize=12)
            ax.set_ylabel('k_perpendicular', fontsize=12)
            ax.set_title('k_z vs k_perpendicular For Protons', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)


            plot_path = output_dir / "02_g_kz_kperp_protons_unmod.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
       
        # Plot 2_7: kz vs k_perp for protons
        if len(self.kz_kperp_protons_mod) > 0 and len(self.kz_kperp_protons_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))


            kz_mod = [point[0] for point in self.kz_kperp_protons_mod]
            kz_unm = [point[0] for point in self.kz_kperp_protons_unmod]


            kperp_mod = [point[1] for point in self.kz_kperp_protons_mod]
            kperp_unm = [point[1] for point in self.kz_kperp_protons_unmod]


            ax.scatter(kz_mod, kperp_mod, s=8, alpha=0.5, label='Modified', color='red')
            ax.scatter(kz_unm, kperp_unm, s=8, alpha=0.5, label='Unmodified', color='blue')


            ax.set_xlabel('k_z', fontsize=12)
            ax.set_ylabel('k_perpendicular', fontsize=12)
            ax.set_title('k_z vs k_perpendicular For Protons', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)


            plot_path = output_dir / "02_g_kz_kperp_protons_mod_unmod.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # Plot 4: Multiplicity Distribution
        if len(self.multiplicity_mod) > 0 and len(self.multiplicity_unm) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            bins = np.linspace(
                min(min(self.multiplicity_mod), min(self.multiplicity_unm)),
                max(max(self.multiplicity_mod), max(self.multiplicity_unm)),
                50
            )
            ax.hist(self.multiplicity_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.multiplicity_unm, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            ax.set_xlabel('Particles per Event', fontsize=12)
            ax.set_ylabel('Event Count', fontsize=12)
            ax.set_title('Event Multiplicity Distribution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plot_path = output_dir / "04_multiplicity_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    
    def get_statistics(self) -> Tuple[Dict]:
        # Perform aggregated detection if not already done
        self._finalize_and_detect()
        
        # Build statistics dictionary
        stats = {
            'total_events_modified': int(self.total_events_mod),
            'total_events_unmodified': int(self.total_events_unm),
            'total_particles_modified': int(self.total_particles_mod),
            'total_particles_unmodified': int(self.total_particles_unm),
            'avg_multiplicity_modified': self.total_particles_mod / max(self.total_events_mod, 1),
            'avg_multiplicity_unmodified': self.total_particles_unm / max(self.total_events_unm, 1),
            'n_angular_samples_modified': len(self.theta_dist_mod),
            'n_angular_samples_unmodified': len(self.theta_dist_unm),
        }
        
        # Return tuple: (stats)
        return stats
    
    
    def _finalize_and_detect(self) -> None:
        if self._finalized:
            return  # Already done
        
        self._finalized = True


    def reset(self) -> None:
        """Reset analyzer state for reuse."""
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        self.multiplicity_mod = []
        self.multiplicity_unm = []
        self.theta_dist_mod = []
        self.theta_dist_unm = []
        self._finalized = False
        self.kz_kperp_charged_mod = []
        self.kz_kperp_charged_unmod = []
        self.kz_kperp_protons_mod = []
        self.kz_kperp_protons_unmod = []
