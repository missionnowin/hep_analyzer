import sys
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np

from models.ions import CHARGED_PDGIDS

try:
    from models.cumulative_singnature import CumulativeSignature
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CumulativeAnalyzer:
    """
    Detect cumulative effect signatures in high-energy physics data.
    """

    def __init__(self, threshold_strength: float = 0.01, threshold_confidence: float = 0.05, 
                 threshold_absolute_excess: int = 0):
        """
        Detect particles scattered to forbidden kinematic regions.
        Also detect anomalies in angular distributions (cumulative effect indicator).
        
        Thresholds:
            threshold_strength (0-1): Fractional excess 
                - 0.05 = flag if >5% more forbidden particles in modified
                - 0.10 = flag if >10% more forbidden particles in modified
                - 0.001 = very sensitive (catch small effects)
                
            threshold_confidence (0-1): Confidence score
                - Higher = more selective
                - confidence = strength * 0.9 if excess > 0
                
            threshold_absolute_excess (int): Minimum absolute particle count
                - 0 = disabled
                - 10 = flag if at least 10 extra forbidden particles
                - Useful for small datasets
        
        Signal fires when: strength > threshold_strength AND confidence > threshold_confidence
        """
        self.threshold_strength = threshold_strength
        self.threshold_confidence = threshold_confidence
        self.threshold_absolute_excess = threshold_absolute_excess
        
        self.total_events_mod = 0
        self.total_events_unm = 0

        self.total_particles_mod = 0
        self.total_particles_unm = 0

        self.total_forbidden_mod = 0
        self.total_forbidden_unm = 0
        
        self.signatures: List[CumulativeSignature] = []
        
        self.multiplicity_mod = []
        self.multiplicity_unm = []
        
        # Statistics
        self.forbidden_reasons_mod = {"very_forward": 0, "very_backward": 0, "impossible_kin": 0, "extreme_sideways": 0}
        self.forbidden_reasons_unm = {"very_forward": 0, "very_backward": 0, "impossible_kin": 0, "extreme_sideways": 0}
        
        # Angular distributions
        self.theta_dist_mod = []
        self.theta_dist_unm = []
        self.phi_dist_mod = []
        self.phi_dist_unm = []

        self.phi_dist_charged_mod = []
        self.phi_dist_charged_unmod = []

        self.theta_dist_charged_mod = []
        self.theta_dist_charged_unmod = []

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
    

    def _accumulate_from_batch(self, batch_mod: List, batch_unm: List) -> None:
        """Accumulate forbidden particles and angular samples from batch."""
        # Accumulate data
        for event in batch_mod:
            if event:
                for p in event:
                    reason = self._get_forbidden_reason(p)
                    theta, phi = self._get_angles_from_particle(p)
                    if theta is not None:
                        self.theta_dist_mod.append(theta)
                        if p.particle_id in CHARGED_PDGIDS:
                            self.theta_dist_charged_mod.append(theta)
                    if phi is not None:
                        self.phi_dist_mod.append(phi)
                        if p.particle_id in CHARGED_PDGIDS:
                            self.phi_dist_charged_mod.append(phi)
                    if reason:
                        self.total_forbidden_mod += 1
                        self.forbidden_reasons_mod[reason] += 1
        
        for event in batch_unm:
            if event:
                for p in event:
                    reason = self._get_forbidden_reason(p)
                    theta, phi = self._get_angles_from_particle(p)
                    if theta is not None:
                        self.theta_dist_unm.append(theta)
                        if p.particle_id in CHARGED_PDGIDS:
                            self.theta_dist_charged_unmod.append(theta)
                    if phi is not None:
                        self.phi_dist_unm.append(phi)
                        if p.particle_id in CHARGED_PDGIDS:
                            self.phi_dist_charged_unmod.append(phi)
                    if reason:
                        self.total_forbidden_unm += 1
                        self.forbidden_reasons_unm[reason] += 1


    
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
    

    def _ks_statistic(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Kolmogorov-Smirnov statistic: maximum difference between two distributions.
        Returns value between 0 (identical) and 1 (completely different).
        """
        sorted1 = np.sort(data1)
        sorted2 = np.sort(data2)
        
        all_values = np.sort(np.concatenate([sorted1, sorted2]))
        cdf1_interp = np.searchsorted(sorted1, all_values, side='right') / len(sorted1)
        cdf2_interp = np.searchsorted(sorted2, all_values, side='right') / len(sorted2)
        
        ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))
        return float(ks_stat)
    

    def _get_forbidden_reason(self, p: Particle) -> str:
        """
        Check if particle is in forbidden kinematic region.
        Returns reason string or None if valid kinematics.
        """
        pt = (p.px**2 + p.py**2) ** 0.5
        p_total = (p.px**2 + p.py**2 + p.pz**2) ** 0.5
        
        if p.pz > 3.0 and pt < 0.1:
            return "very_forward"
        if p.pz < -3.0 and pt < 0.2:
            return "very_backward"
        if pt > 2.0 and p_total < 0.5:
            return "impossible_kin"
        if pt > 1.5 and abs(p.pz) < 0.1:
            return "extreme_sideways"
        
        return None

    
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
            
            ax.set_xlabel('Polar Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "01_theta_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)

        # Plot 2: Angular Distribution (Theta)
        if len(self.theta_dist_charged_mod) > 0 and len(self.theta_dist_charged_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bins = np.linspace(0, np.pi, 50)
            ax.hist(self.theta_dist_charged_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.theta_dist_charged_unmod, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            
            ax.set_xlabel('Polar Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison For Charged Particles', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "02_theta_distribution_charged.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)

        # Plot 3: Angular Distribution (Phi)
        if len(self.phi_dist_mod) > 0 and len(self.phi_dist_unm) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bins = np.linspace(0, np.pi, 50)
            ax.hist(self.phi_dist_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.phi_dist_unm, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            
            ax.set_xlabel('Azimutal Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "03_phi_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # Plot 4: Angular Distribution (Phi)
        if len(self.phi_dist_charged_mod) > 0 and len(self.phi_dist_charged_unmod) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bins = np.linspace(0, np.pi, 50)
            ax.hist(self.phi_dist_charged_mod, bins=bins, alpha=0.6, label='Modified', color='red', edgecolor='black')
            ax.hist(self.phi_dist_charged_unmod, bins=bins, alpha=0.6, label='Unmodified', color='blue', edgecolor='black')
            
            ax.set_xlabel('Azimutal Angle θ (radians)', fontsize=12)
            ax.set_ylabel('Particle Count', fontsize=12)
            ax.set_title('Angular Distribution Comparison For Charged Particles', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plot_path = output_dir / "04_phi_distribution_charged.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # Plot 5: Forbidden Particles Breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = list(self.forbidden_reasons_mod.keys())
        mod_counts = [self.forbidden_reasons_mod[cat] for cat in categories]
        unm_counts = [self.forbidden_reasons_unm[cat] for cat in categories]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, mod_counts, width, label='Modified', color='red', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, unm_counts, width, label='Unmodified', color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Forbidden Region Type', fontsize=12)
        ax.set_ylabel('Particle Count', fontsize=12)
        ax.set_title('Forbidden Kinematic Regions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plot_path = output_dir / "05_forbidden_regions.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # Plot 6: Multiplicity Distribution
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
            plot_path = output_dir / "06_multiplicity_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # Plot 7: Signature Summary
        if len(self.signatures) > 0:
            sig_types = {}
            sig_strengths = {}
            for sig in self.signatures:
                sig_type = sig.signature_type
                sig_types[sig_type] = sig_types.get(sig_type, 0) + 1
                if sig_type not in sig_strengths:
                    sig_strengths[sig_type] = []
                sig_strengths[sig_type].append(sig.strength)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            types = list(sig_types.keys())
            counts = list(sig_types.values())
            colors = ['red' if 'forbidden' in t else 'green' for t in types]
            ax1.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Count', fontsize=11)
            ax1.set_title('Detected Signatures by Type', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            for sig_type, strengths in sig_strengths.items():
                ax2.hist(strengths, alpha=0.6, label=sig_type, bins=20, edgecolor='black')
            ax2.set_xlabel('Signature Strength', fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.set_title('Signature Strength Distribution', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plot_path = output_dir / "07_signature_summary.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    
    def get_statistics(self) -> Tuple[Dict, List[CumulativeSignature]]:
        """
        UNIFIED METHOD: Finalize detection AND return both statistics and signatures.
        
        Returns:
            Tuple of (statistics_dict, signatures_list)
        
        This is the only method you need to call after processing all batches.
        It performs finalization internally and returns everything.
        
        Usage:
            stats, sigs = analyzer.get_statistics()
            print(f"Detected {len(sigs)} signatures")
            print(f"Total events: {stats['total_events_modified']}")
        """
        # Perform aggregated detection if not already done
        self._finalize_and_detect()
        
        # Build statistics dictionary
        stats = {
            'total_events_modified': int(self.total_events_mod),
            'total_events_unmodified': int(self.total_events_unm),
            'total_particles_modified': int(self.total_particles_mod),
            'total_particles_unmodified': int(self.total_particles_unm),
            'signatures_detected': len(self.signatures),
            'avg_multiplicity_modified': self.total_particles_mod / max(self.total_events_mod, 1),
            'avg_multiplicity_unmodified': self.total_particles_unm / max(self.total_events_unm, 1),
            'forbidden_breakdown_modified': self.forbidden_reasons_mod,
            'forbidden_breakdown_unmodified': self.forbidden_reasons_unm,
            'total_forbidden_modified': self.total_forbidden_mod,
            'total_forbidden_unmodified': self.total_forbidden_unm,
            'n_angular_samples_modified': len(self.theta_dist_mod),
            'n_angular_samples_unmodified': len(self.theta_dist_unm),
        }
        
        # Return tuple: (stats, signatures)
        return stats, self.signatures
    
    
    def _finalize_and_detect(self) -> None:
        """
        INTERNAL: Perform aggregated analysis to detect signatures.
        Called once from get_statistics() if not already finalized.
        """
        if self._finalized:
            return  # Already done
        
        self._detect_forbidden_kinematics_aggregated()
        self._detect_angular_distribution_anomaly_aggregated()
        
        self._finalized = True
    

    def _detect_forbidden_kinematics_aggregated(self) -> None:
        """Analyze forbidden particles across ENTIRE dataset."""
        if self.total_forbidden_mod == 0 and self.total_forbidden_unm == 0:
            return
        
        excess = self.total_forbidden_mod - self.total_forbidden_unm
        
        if self.total_forbidden_unm > 0:
            strength = min(1.0, excess / max(self.total_forbidden_unm, 1))
        elif self.total_forbidden_mod > 0:
            strength = 1.0
        else:
            strength = 0.0
        
        confidence = strength * 0.9 if excess > 0 else 0.0
        
        should_fire = (
            strength > self.threshold_strength and 
            confidence > self.threshold_confidence
        )
        
        if self.threshold_absolute_excess > 0 and excess >= self.threshold_absolute_excess:
            should_fire = True
        
        if should_fire:
            reasons_mod = ", ".join([f"{k}={v}" for k, v in self.forbidden_reasons_mod.items() if v > 0])
            reasons_unm = ", ".join([f"{k}={v}" for k, v in self.forbidden_reasons_unm.items() if v > 0]) or "none"
            
            sig = CumulativeSignature(
                signature_type="forbidden_kinematics_aggregated",
                strength=float(strength),
                confidence=float(confidence),
                affected_particles=int(excess),
                description=f"AGGREGATED: mod=({reasons_mod}) unm=({reasons_unm}) excess={excess}, total_ratio={self.total_forbidden_mod}/{self.total_forbidden_unm}"
            )
            self.signatures.append(sig)
    

    def _detect_angular_distribution_anomaly_aggregated(self) -> None:
        """Analyze angular distributions across ENTIRE dataset."""
        if len(self.theta_dist_mod) < 10 or len(self.theta_dist_unm) < 10:
            return
        
        thetas_mod = np.array(self.theta_dist_mod)
        thetas_unm = np.array(self.theta_dist_unm)
        
        ks_statistic = self._ks_statistic(thetas_mod, thetas_unm)
        strength = min(1.0, ks_statistic * 2.0)
        confidence = strength * 0.8 if ks_statistic > 0.05 else 0.0
        
        if strength > self.threshold_strength and confidence > self.threshold_confidence:
            mean_theta_mod = np.mean(thetas_mod)
            mean_theta_unm = np.mean(thetas_unm)
            std_theta_mod = np.std(thetas_mod)
            std_theta_unm = np.std(thetas_unm)
            
            sig = CumulativeSignature(
                signature_type="angular_distribution_aggregated",
                strength=float(strength),
                confidence=float(confidence),
                affected_particles=len(thetas_mod),
                description=f"AGGREGATED ANGLES: mod_mean={mean_theta_mod:.3f}±{std_theta_mod:.3f}, unm_mean={mean_theta_unm:.3f}±{std_theta_unm:.3f}, KS={ks_statistic:.4f}, samples_mod={len(thetas_mod)}, samples_unm={len(thetas_unm)}"
            )
            self.signatures.append(sig)


    def reset(self) -> None:
        """Reset analyzer state for reuse."""
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        self.signatures = []
        self.multiplicity_mod = []
        self.multiplicity_unm = []
        self.forbidden_reasons_mod = {"very_forward": 0, "very_backward": 0, "impossible_kin": 0, "extreme_sideways": 0}
        self.forbidden_reasons_unm = {"very_forward": 0, "very_backward": 0, "impossible_kin": 0, "extreme_sideways": 0}
        self.theta_dist_mod = []
        self.theta_dist_unm = []
        self.phi_dist_mod = []
        self.phi_dist_unm = []
        self.total_forbidden_mod = 0
        self.total_forbidden_unm = 0
        self._finalized = False