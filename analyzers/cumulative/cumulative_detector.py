import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
M_NUCLEON = 0.938  # GeV (nucleon mass)
M_PION = 0.140     # GeV (for reference)

try:
    from models.cumulative_singnature import CumulativeSignature
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class CumulativeEffectDetector:
    """Detect cumulative scattering effects in particle data
    
    Analyzes both modified (with cumulative effects) and unmodified particles
    to identify signatures of cumulative interactions.
    
    Usage:
        detector = CumulativeEffectDetector(particles_modified, particles_unmodified)
        detector.detect_all_signatures()
        likelihood = detector.get_cumulative_likelihood()
        for sig in detector.signatures:
            print(f"Effect: {sig.signature_type}, Strength: {sig.strength:.2f}")
            print(f"  {sig.description}")
    """
    
    def __init__(self, particles_modified: Optional[List[Particle]] = None,
                 particles_unmodified: Optional[List[Particle]] = None):
        """
        Initialize detector with modified and unmodified particle samples
        
        Args:
            particles_modified: Particles after cumulative interactions
            particles_unmodified: Baseline particles without cumulative effects
        """
        self.particles_modified = particles_modified or []
        self.particles_unmodified = particles_unmodified or []
        self.signatures: List[CumulativeSignature] = []
        self._cumulative_likelihood = 0.0
    
    def detect_all_signatures(self) -> List[CumulativeSignature]:
        """Run all cumulative effect detection methods
        
        Returns:
            List of detected signatures
        """
        self.signatures = []
        
        if not self.particles_modified:
            return self.signatures
        
        # Run all detection methods
        self._detect_angular_deflection()
        self._detect_energy_loss()
        self._detect_spectrum_modification()
        self._detect_forward_backward_asymmetry()
        self._detect_pt_suppression()
        
        # Calculate overall likelihood
        self._calculate_cumulative_likelihood()
        
        return self.signatures
    
    def _detect_angular_deflection(self) -> None:
        """Detect angular deflection of high-pT particles
        
        Cumulative scattering causes particles to deviate from original direction.
        Look for particles with large angle changes.
        """
        if not self.particles_modified or not self.particles_unmodified:
            return
        
        if len(self.particles_modified) != len(self.particles_unmodified):
            return
        
        # Calculate angles for modified particles
        theta_mod = np.array([
            self._calculate_polar_angle(p) for p in self.particles_modified
        ], dtype=np.float32)
        
        # Calculate angles for unmodified particles
        theta_unm = np.array([
            self._calculate_polar_angle(p) for p in self.particles_unmodified
        ], dtype=np.float32)
        
        # Calculate angle differences
        delta_theta = np.abs(theta_mod - theta_unm)
        delta_theta_deg = np.degrees(delta_theta)
        
        # Select high-pT particles
        pt_mod = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_modified
        ], dtype=np.float32)
        
        high_pt_mask = pt_mod > 1.0 # GeV
        
        if high_pt_mask.sum() > 0:
            mean_deflection = np.mean(delta_theta_deg[high_pt_mask])
            std_deflection = np.std(delta_theta_deg[high_pt_mask])

            # Strength: normalized by typical deflection (~0.5 degrees for significant effect)
            strength = min(1.0, mean_deflection / 0.5)
            affected = int(high_pt_mask.sum())

            # Confidence based on statistical significance
            if std_deflection > 0 and affected > 10:
                confidence = min(1.0, (mean_deflection / std_deflection) / 3.0)
            else:
                confidence = 0.3
            
            if strength > 0.05: # Only report if detectable
                sig = CumulativeSignature(
                    signature_type="angular_deflection",
                    strength=strength,
                    confidence=confidence,
                    affected_particles=affected,
                    description=f"High-pT particles deflected by {mean_deflection:.2f}° (±{std_deflection:.2f}°)"
                )
                self.signatures.append(sig)
    
    def _detect_energy_loss(self) -> None:
        """Detect energy loss in modified particles
        
        Cumulative interactions cause particles to lose energy traversing medium.
        Compare total momentum distributions.
        """
        if not self.particles_modified or not self.particles_unmodified:
            return
        
        # Calculate total momentum
        p_mod = np.array([
            np.sqrt(p.px**2 + p.py**2 + p.pz**2) for p in self.particles_modified
        ], dtype=np.float32)
        
        p_unm = np.array([
            np.sqrt(p.px**2 + p.py**2 + p.pz**2) for p in self.particles_unmodified
        ], dtype=np.float32)
        
        # Calculate mean momentum
        mean_p_mod = np.mean(p_mod[p_mod > 0.1])
        mean_p_unm = np.mean(p_unm[p_unm > 0.1])
        
        if mean_p_unm > 0:
            energy_loss_fraction = (mean_p_unm - mean_p_mod) / mean_p_unm
            
            # Only consider as signature if loss is significant (>5%)
            if energy_loss_fraction > 0.05:
                strength = min(1.0, energy_loss_fraction / 0.3)

                # Confidence based on how consistent the loss is
                p_ratio = p_mod[p_mod > 0.1] / mean_p_unm
                consistency = 1.0 - np.std(p_ratio)
                confidence = min(1.0, max(0.3, consistency))
                
                sig = CumulativeSignature(
                    signature_type="energy_loss",
                    strength=strength,
                    confidence=confidence,
                    affected_particles=len(self.particles_modified),
                    description=f"Average momentum loss: {energy_loss_fraction*100:.1f}%"
                )
                self.signatures.append(sig)
    
    def _detect_spectrum_modification(self) -> None:
        """Detect modification of pT spectrum
        
        Cumulative effects typically suppress high-pT particles.
        """
        if not self.particles_modified or not self.particles_unmodified:
            return
        
        # Calculate pT distributions
        pt_mod = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_modified
        ], dtype=np.float32)
        
        pt_unm = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_unmodified
        ], dtype=np.float32)
        
        # High-pT suppression: ratio of high-pT particles
        high_pt_threshold = 2.0  # GeV
        high_pt_ratio_mod = (pt_mod > high_pt_threshold).sum() / len(pt_mod)
        high_pt_ratio_unm = (pt_unm > high_pt_threshold).sum() / len(pt_unm)
        
        if high_pt_ratio_unm > 0.01:
            suppression = (high_pt_ratio_unm - high_pt_ratio_mod) / high_pt_ratio_unm
            
            if suppression > 0.05:
                strength = min(1.0, suppression / 0.5)

                # Confidence from number of affected particles
                n_affected = int((pt_mod > high_pt_threshold).sum())
                confidence = min(1.0, 0.3 + (n_affected / 100.0) * 0.7)
                
                sig = CumulativeSignature(
                    signature_type="spectrum_modification",
                    strength=strength,
                    confidence=confidence,
                    affected_particles=n_affected,
                    description=f"High-pT suppression (pT>{high_pt_threshold} GeV): {suppression*100:.1f}%"
                )
                self.signatures.append(sig)
    
    def _detect_forward_backward_asymmetry(self) -> None:
        """Detect forward-backward asymmetry
        
        Cumulative effects often create asymmetry between forward and backward hemispheres.
        """
        if not self.particles_modified:
            return
        
        # Calculate pseudorapidity
        eta = np.array([
            self._calculate_pseudorapidity(p) for p in self.particles_modified
        ], dtype=np.float32)
        
        # Forward vs backward particles
        forward_mask = eta > 0
        backward_mask = eta < 0
        
        n_forward = forward_mask.sum()
        n_backward = backward_mask.sum()
        n_total = len(eta)
        
        if n_total > 20:
            forward_fraction = n_forward / n_total
            backward_fraction = n_backward / n_total

            # Expected ~50% in each hemisphere without asymmetry
            asymmetry = abs(forward_fraction - backward_fraction)
            
            if asymmetry > 0.05:
                strength = min(1.0, asymmetry / 0.3)

                # Confidence from particle count
                confidence = min(1.0, 0.3 + (n_total / 200.0) * 0.7)
                
                sig = CumulativeSignature(
                    signature_type="forward_backward_asymmetry",
                    strength=strength,
                    confidence=confidence,
                    affected_particles=n_total,
                    description=f"Forward: {forward_fraction*100:.1f}%, Backward: {backward_fraction*100:.1f}%"
                )
                self.signatures.append(sig)
    
    def _detect_pt_suppression(self) -> None:
        """Detect transverse momentum suppression
        
        High-pT particles are preferentially suppressed by cumulative interactions.
        """
        if not self.particles_modified or not self.particles_unmodified:
            return
        
        pt_mod = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_modified
        ], dtype=np.float32)
        
        pt_unm = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_unmodified
        ], dtype=np.float32)
        
        # Compare high-pT particles in midrapidity region
        eta_mod = np.array([
            self._calculate_pseudorapidity(p) for p in self.particles_modified
        ], dtype=np.float32)
        
        eta_unm = np.array([
            self._calculate_pseudorapidity(p) for p in self.particles_unmodified
        ], dtype=np.float32)
        
        # Select midrapidity
        mid_mask_mod = np.abs(eta_mod) < 1.0
        mid_mask_unm = np.abs(eta_unm) < 1.0
        
        if mid_mask_mod.sum() > 10 and mid_mask_unm.sum() > 10:
            # Mean pT in midrapidity
            mean_pt_mod = np.mean(pt_mod[mid_mask_mod])
            mean_pt_unm = np.mean(pt_unm[mid_mask_unm])
            
            if mean_pt_unm > 0:
                suppression = (mean_pt_unm - mean_pt_mod) / mean_pt_unm
                
                if suppression > 0.03:
                    strength = min(1.0, suppression / 0.2)
                    n_affected = int(mid_mask_mod.sum())
                    confidence = min(1.0, 0.5 + (suppression / 0.2) * 0.5)
                    
                    sig = CumulativeSignature(
                        signature_type="pt_suppression",
                        strength=strength,
                        confidence=confidence,
                        affected_particles=n_affected,
                        description=f"Midrapidity pT reduction: {suppression*100:.1f}% (|η|<1.0)"
                    )
                    self.signatures.append(sig)
    
    def _calculate_cumulative_likelihood(self) -> None:
        """Calculate overall cumulative effect likelihood
        
        Combines all detected signatures into single likelihood value.
        """
        if not self.signatures:
            self._cumulative_likelihood = 0.0
            return
        
        # Weight each signature by its strength and confidence
        weighted_strengths = [
            sig.strength * sig.confidence for sig in self.signatures
        ]
        
        # Average of weighted signatures, capped at 1.0
        self._cumulative_likelihood = min(1.0, np.mean(weighted_strengths))
    
    def get_cumulative_likelihood(self) -> float:
        """Get overall likelihood of cumulative effects (0-1)
        
        Returns:
            Likelihood value where 0 = no effects, 1 = strong cumulative effects
        """
        return self._cumulative_likelihood
    
    def cumulative_candidate_mask(
        self,
        pt_min: float = 1.0,
        y_window: Tuple[float, float] = (-0.5, 0.5),
    ) -> np.ndarray:
        """Identify particles likely affected by cumulative scattering
        
        High-pT particles in midrapidity are most affected by cumulative effects.
        
        Args:
            pt_min: Minimum transverse momentum (GeV)
            y_window: Rapidity window (y_min, y_max)
            
        Returns:
            Boolean mask for candidate particles
        """
        if not self.particles_modified:
            return np.array([], dtype=bool)
        
        pt_vals = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_modified
        ], dtype=np.float32)
        
        y_vals = np.array([
            self._calculate_rapidity(p) for p in self.particles_modified
        ], dtype=np.float32)
        
        mask = (pt_vals >= pt_min) & (y_vals >= y_window) & (y_vals <= y_window)
        return mask
    
    def plot_cumulative_candidates(
        self,
        out: Optional[str] = None,
        pt_min: float = 1.0,
        y_window: Tuple[float, float] = (-0.5, 0.5)
    ) -> None:
        """Plot particles identified as cumulative effect candidates
        
        Args:
            out: Output file path
            pt_min: Minimum pT threshold (GeV)
            y_window: Rapidity window
        """
        mask = self.cumulative_candidate_mask(pt_min, y_window)
        
        if not np.any(mask):
            print(f"[WARNING] No cumulative candidates found (pT>{pt_min}, {y_window}<y<{y_window})")
            return
        
        pt_vals = np.array([
            np.sqrt(p.px**2 + p.py**2) for p in self.particles_modified
        ], dtype=np.float32)
        
        y_vals = np.array([
            self._calculate_rapidity(p) for p in self.particles_modified
        ], dtype=np.float32)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            y_vals[mask], pt_vals[mask],
            s=30, edgecolor="black", facecolor="red", alpha=0.7,
            label=f"Cumulative candidates (N={mask.sum()})"
        )
        
        # Also show all particles for context
        plt.scatter(
            y_vals[~mask], pt_vals[~mask],
            s=10, edgecolor="gray", facecolor="lightgray", alpha=0.3,
            label="Other particles"
        )
        
        plt.xlabel("Rapidity y", fontsize=12)
        plt.ylabel(r"$p_T$ (GeV)", fontsize=12)
        plt.title(f"Cumulative Effect Candidates\n(pT>{pt_min} GeV, {y_window}<y<{y_window})", fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.axhline(pt_min, color="red", linestyle="--", linewidth=1, alpha=0.5)
        plt.tight_layout()
        
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[INFO] Cumulative candidates plot saved to {out}")
        
        plt.close()
    
    def plot_cumulative_summary(self, out: Optional[str] = None) -> None:
        """Plot summary of all detected cumulative signatures
        
        Args:
            out: Output file path
        """
        if not self.signatures:
            print("[WARNING] No signatures detected. Run detect_all_signatures() first.")
            return
        
        # Extract data
        types = [sig.signature_type for sig in self.signatures]
        strengths = [sig.strength for sig in self.signatures]
        confidences = [sig.confidence for sig in self.signatures]
        affected = [sig.affected_particles for sig in self.signatures]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Cumulative Effects Summary", fontsize=14, fontweight="bold")
        
        # 1. Signature strengths
        ax = axes[0, 0]
        x = np.arange(len(types))
        bars = ax.bar(x, strengths, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_ylabel("Strength", fontsize=11)
        ax.set_title("Effect Strength by Type", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, strengths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        
        # 2. Confidence levels
        ax = axes[0, 1]
        bars = ax.bar(x, confidences, color="darkgreen", edgecolor="black", alpha=0.7)
        ax.set_ylabel("Confidence", fontsize=11)
        ax.set_title("Detection Confidence by Type", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        
        # 3. Affected particles
        ax = axes[1, 0]
        bars = ax.bar(x, affected, color="teal", edgecolor="black", alpha=0.7)
        ax.set_ylabel("Number of Particles", fontsize=11)
        ax.set_title("Affected Particles by Type", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, affected):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{int(val)}", ha="center", va="bottom", fontsize=9)
        
        # 4. Overall likelihood
        ax = axes[1, 1]
        ax.axis("off")
        
        overall_text = f"Overall Cumulative Likelihood: {self.get_cumulative_likelihood():.3f}"
        ax.text(0.5, 0.7, overall_text, ha="center", va="center", fontsize=14,
               fontweight="bold", transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
        
        # Summary text
        summary_lines = [
            f"Total signatures detected: {len(self.signatures)}",
            f"Mean strength: {np.mean(strengths):.3f}",
            f"Mean confidence: {np.mean(confidences):.3f}",
            f"Total affected particles: {sum(affected)}",
        ]
        
        summary_text = "\n".join(summary_lines)
        ax.text(0.5, 0.3, summary_text, ha="center", va="center", fontsize=11,
               transform=ax.transAxes, family="monospace",
               bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[INFO] Cumulative summary plot saved to {out}")
        
        plt.close()
    
    # -------- Static helper methods --------
    
    @staticmethod
    def _calculate_polar_angle(particle: Particle) -> float:
        """Calculate polar angle theta (0 to pi radians)"""
        pt = np.sqrt(particle.px**2 + particle.py**2)
        theta = np.arctan2(pt, particle.pz)
        return theta
    
    @staticmethod
    def _calculate_pseudorapidity(particle: Particle) -> float:
        """Calculate pseudorapidity eta"""
        pt = np.sqrt(particle.px**2 + particle.py**2)
        theta = np.arctan2(pt, particle.pz)

        # Avoid singularity
        theta = np.clip(theta, 1e-8, np.pi - 1e-8)
        eta = -np.log(np.tan(theta / 2.0))
        return eta
    
    @staticmethod
    def _calculate_rapidity(particle: Particle) -> float:
        """Calculate rapidity y"""
        E = particle.E
        pz = particle.pz
        
        # Avoid singularity
        if abs(E - pz) < 1e-8:
            return 0.0
        
        y = 0.5 * np.log((E + pz) / (E - pz))
        return y
