import sys
import json
from typing import Dict, List
from pathlib import Path

import numpy as np

try:
    from models.cumulative_singnature import CumulativeSignature
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CumulativeAnalyzer:
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
        
        self.signatures = []
        
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
    
    def process_batch(self, batch_mod: List, batch_unm: List) -> None:
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
        
        self._detect_signatures_from_batch(batch_mod, batch_unm)
    
    def _detect_signatures_from_batch(self, batch_mod: List, batch_unm: List) -> None:
        if not batch_mod or not batch_unm:
            return
        
        self._detect_forbidden_kinematics(batch_mod, batch_unm)
        self._detect_angular_distribution_anomaly(batch_mod, batch_unm)
    
    def _get_angles_from_particle(self, p) -> tuple:
        """
        Calculate theta (polar angle) and phi (azimuthal angle) from particle momentum.
        Returns (theta, phi) in radians
        """
        pt = (p.px**2 + p.py**2) ** 0.5
        p_total = (p.px**2 + p.py**2 + p.pz**2) ** 0.5
        
        # Avoid division by zero
        if p_total == 0:
            return None, None
        
        # Polar angle: theta = arccos(pz / p_total)
        theta = np.arccos(np.clip(p.pz / p_total, -1.0, 1.0))
        
        # Azimuthal angle: phi = atan2(py, px)
        phi = np.arctan2(p.py, p.px)
        
        return theta, phi
    
    def _detect_angular_distribution_anomaly(self, batch_mod: List, batch_unm: List) -> None:
        """
        Detect anomalies in angular distributions (theta, phi).
        Cumulative effects scatter particles, distorting the normal distribution.
        """
        thetas_mod = []
        thetas_unm = []
        
        for event in batch_mod:
            if event:
                for p in event:
                    theta, phi = self._get_angles_from_particle(p)
                    if theta is not None:
                        thetas_mod.append(theta)
                        self.theta_dist_mod.append(theta)
        
        for event in batch_unm:
            if event:
                for p in event:
                    theta, phi = self._get_angles_from_particle(p)
                    if theta is not None:
                        thetas_unm.append(theta)
                        self.theta_dist_unm.append(theta)
        
        # Need enough particles to compare distributions
        if len(thetas_mod) < 10 or len(thetas_unm) < 10:
            return
        
        thetas_mod = np.array(thetas_mod)
        thetas_unm = np.array(thetas_unm)
        
        # Compare distributions using KS test (statistical distance)
        # KS statistic: maximum difference between CDFs
        ks_statistic = self._ks_statistic(thetas_mod, thetas_unm)
        
        # Strength = how different the distributions are
        strength = min(1.0, ks_statistic * 2.0)  # Scale to 0-1
        confidence = strength * 0.8 if ks_statistic > 0.05 else 0.0
        
        if strength > self.threshold_strength and confidence > self.threshold_confidence:
            # Calculate some summary stats
            mean_theta_mod = np.mean(thetas_mod)
            mean_theta_unm = np.mean(thetas_unm)
            std_theta_mod = np.std(thetas_mod)
            std_theta_unm = np.std(thetas_unm)
            
            sig = CumulativeSignature(
                signature_type="angular_distribution",
                strength=float(strength),
                confidence=float(confidence),
                affected_particles=len(thetas_mod),
                description=f"Angle distortion: mod_mean={mean_theta_mod:.3f}±{std_theta_mod:.3f}, unm_mean={mean_theta_unm:.3f}±{std_theta_unm:.3f}, KS={ks_statistic:.4f}"
            )
            self.signatures.append(sig)
    
    def _ks_statistic(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute Kolmogorov-Smirnov statistic: maximum difference between two distributions.
        Returns value between 0 (identical) and 1 (completely different).
        """
        # Sort both datasets
        sorted1 = np.sort(data1)
        sorted2 = np.sort(data2)
        
        # Compute empirical CDFs
        cdf1 = np.arange(1, len(sorted1) + 1) / len(sorted1)
        cdf2 = np.arange(1, len(sorted2) + 1) / len(sorted2)
        
        # Interpolate to find max difference
        all_values = np.sort(np.concatenate([sorted1, sorted2]))
        cdf1_interp = np.searchsorted(sorted1, all_values, side='right') / len(sorted1)
        cdf2_interp = np.searchsorted(sorted2, all_values, side='right') / len(sorted2)
        
        ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))
        return float(ks_stat)
    
    def _get_forbidden_reason(self, p) -> str:
        """
        Check if particle is in forbidden kinematic region.
        Returns reason or None.
        """
        pt = (p.px**2 + p.py**2) ** 0.5
        p_total = (p.px**2 + p.py**2 + p.pz**2) ** 0.5
        
        # Very forward: high pz, very low pt (unphysical)
        if p.pz > 3.0 and pt < 0.1:
            return "very_forward"
        
        # Very backward: large negative pz
        if p.pz < -3.0 and pt < 0.2:
            return "very_backward"
        
        # High pT with very low total momentum (impossible)
        if pt > 2.0 and p_total < 0.5:
            return "impossible_kin"
        
        # Extreme sideways: high pt with low pz
        if pt > 1.5 and abs(p.pz) < 0.1:
            return "extreme_sideways"
        
        return None
    
    def _is_forbidden_kinematic(self, p) -> bool:
        """Check if particle is in forbidden kinematic region."""
        return self._get_forbidden_reason(p) is not None
    
    def _detect_forbidden_kinematics(self, batch_mod: List, batch_unm: List) -> None:
        """
        Detect particles scattered to forbidden kinematic regions.
        Signature = MORE forbidden particles in modified than unmodified.
        """
        forbidden_mod = 0
        forbidden_unm = 0
        batch_reasons_mod = {"very_forward": 0, "very_backward": 0, "impossible_kin": 0, "extreme_sideways": 0}
        batch_reasons_unm = {"very_forward": 0, "very_backward": 0, "impossible_kin": 0, "extreme_sideways": 0}
        
        for event in batch_mod:
            if event:
                for p in event:
                    reason = self._get_forbidden_reason(p)
                    if reason:
                        forbidden_mod += 1
                        batch_reasons_mod[reason] += 1
                        self.forbidden_reasons_mod[reason] += 1
        
        for event in batch_unm:
            if event:
                for p in event:
                    reason = self._get_forbidden_reason(p)
                    if reason:
                        forbidden_unm += 1
                        batch_reasons_unm[reason] += 1
                        self.forbidden_reasons_unm[reason] += 1
        
        # Calculate strength and confidence
        excess = forbidden_mod - forbidden_unm
        
        if forbidden_unm > 0:
            strength = min(1.0, excess / max(forbidden_unm, 1))
        elif forbidden_mod > 0:
            strength = 1.0
        else:
            strength = 0.0
        
        confidence = strength * 0.9 if excess > 0 else 0.0
        
        # Fire signal if BOTH thresholds met
        should_fire = (
            strength > self.threshold_strength and 
            confidence > self.threshold_confidence
        )
        
        # Optional: also fire if absolute excess threshold met
        if self.threshold_absolute_excess > 0 and excess >= self.threshold_absolute_excess:
            should_fire = True
        
        if should_fire:
            reasons_mod = ", ".join([f"{k}={v}" for k, v in batch_reasons_mod.items() if v > 0])
            reasons_unm = ", ".join([f"{k}={v}" for k, v in batch_reasons_unm.items() if v > 0]) or "none"
            sig = CumulativeSignature(
                signature_type="forbidden_kinematics",
                strength=float(strength),
                confidence=float(confidence),
                affected_particles=int(excess),
                description=f"mod=({reasons_mod}) unm=({reasons_unm}) excess={excess}"
            )
            self.signatures.append(sig)
    
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
        
        # Plot 2: Forbidden Particles Breakdown
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
        
        plot_path = output_dir / "02_forbidden_regions.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # Plot 3: Multiplicity Distribution
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
            
            plot_path = output_dir / "03_multiplicity_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # Plot 4: Signature Summary (if signatures detected)
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
            
            # Signature count
            types = list(sig_types.keys())
            counts = list(sig_types.values())
            colors = ['red' if 'forbidden' in t else 'green' for t in types]
            ax1.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Count', fontsize=11)
            ax1.set_title('Detected Signatures by Type', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Signature strength distribution
            for sig_type, strengths in sig_strengths.items():
                ax2.hist(strengths, alpha=0.6, label=sig_type, bins=20, edgecolor='black')
            ax2.set_xlabel('Signature Strength', fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.set_title('Signature Strength Distribution', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plot_path = output_dir / "04_signature_summary.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def get_signatures(self) -> List[CumulativeSignature]:
        return self.signatures
    
    def get_statistics(self) -> Dict:
        return {
            'total_events_modified': int(self.total_events_mod),
            'total_events_unmodified': int(self.total_events_unm),
            'total_particles_modified': int(self.total_particles_mod),
            'total_particles_unmodified': int(self.total_particles_unm),
            'signatures_detected': len(self.signatures),
            'avg_multiplicity_modified': self.total_particles_mod / max(self.total_events_mod, 1),
            'avg_multiplicity_unmodified': self.total_particles_unm / max(self.total_events_unm, 1),
            'forbidden_breakdown_modified': self.forbidden_reasons_mod,
            'forbidden_breakdown_unmodified': self.forbidden_reasons_unm,
            'n_angular_samples_modified': len(self.theta_dist_mod),
            'n_angular_samples_unmodified': len(self.theta_dist_unm),
        }
    
    def reset(self) -> None:
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