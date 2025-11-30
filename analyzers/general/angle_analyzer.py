import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
import json

try:
    from utils.readers import OscarFormat, OscarReader
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class AngleAnalyzer:
    """Calculate and analyze angle distributions"""
    
    @staticmethod
    def polar_angle(particle: Particle) -> float:
        """Calculate polar angle theta (0 to pi radians)
        Angle relative to beam axis (z-axis)
        """
        pt = np.sqrt(particle.px**2 + particle.py**2)
        theta = np.arctan2(pt, particle.pz)
        return theta
    
    @staticmethod
    def azimuthal_angle(particle: Particle) -> float:
        """Calculate azimuthal angle phi (0 to 2pi radians)
        Angle in transverse plane
        """
        phi = np.arctan2(particle.py, particle.px)
        if phi < 0:
            phi += 2 * np.pi
        return phi
    
    @staticmethod
    def transverse_momentum(particle: Particle) -> float:
        """Calculate transverse momentum magnitude"""
        return np.sqrt(particle.px**2 + particle.py**2)
    
    @staticmethod
    def rapidity(particle: Particle) -> Optional[float]:
        """Calculate rapidity y = 0.5 * ln((E+pz)/(E-pz))"""
        numerator = particle.E + particle.pz
        denominator = particle.E - particle.pz
        if denominator > 0:
            return 0.5 * np.log(numerator / denominator)
        return None
    
    @staticmethod
    def pseudorapidity(particle: Particle) -> float:
        """Calculate pseudorapidity eta = -ln(tan(theta/2))"""
        pt = np.sqrt(particle.px**2 + particle.py**2)
        theta = np.arctan2(pt, particle.pz)
        
        # Avoid division by zero
        if np.sin(theta / 2) == 0:
            return 0.0
        
        eta = -np.log(np.tan(theta / 2))
        return eta


class AngleDistributionPlotter:
    """Visualize angle distributions"""
    
    def __init__(self, particles: List[Particle], figsize: Tuple = (14, 10)):
        self.particles = particles
        self.figsize = figsize
        self.analyzer = AngleAnalyzer()
        
    def plot_distributions(self, output_file: Optional[str] = None) -> None:
        """Create comprehensive angle distribution plots"""
        
        # Calculate angles for all particles
        polar_angles = [self.analyzer.polar_angle(p) for p in self.particles]
        azimuthal_angles = [self.analyzer.azimuthal_angle(p) for p in self.particles]
        pt = [self.analyzer.transverse_momentum(p) for p in self.particles]
        
        # Convert to degrees
        polar_deg = np.degrees(polar_angles)
        azimuthal_deg = np.degrees(azimuthal_angles)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('UrQMD Angle Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Polar angle histogram
        ax = axes[0, 0]
        ax.hist(polar_deg, bins=36, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Polar Angle θ (degrees)', fontsize=11)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('Polar Angle Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 180)
        
        # Add statistics
        mean_theta = np.mean(polar_deg)
        ax.axvline(mean_theta, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_theta:.1f}°')
        ax.legend()
        
        # 2. Azimuthal angle histogram
        ax = axes[0, 1]
        ax.hist(azimuthal_deg, bins=36, color='darkgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('Azimuthal Angle Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 360)
        
        # 3. 2D scatter: polar vs azimuthal
        ax = axes[1, 0]
        scatter = ax.scatter(azimuthal_deg, polar_deg, c=pt, cmap='viridis', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
        ax.set_ylabel('Polar Angle θ (degrees)', fontsize=11)
        ax.set_title('Polar vs Azimuthal Angles (colored by pT)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Transverse Momentum (GeV)', fontsize=10)
        ax.grid(alpha=0.3)
        
        # 4. Polar angle with cos(theta) for physics insight
        ax = axes[1, 1]
        cos_theta = np.cos(np.radians(polar_deg))
        ax.hist(cos_theta, bins=40, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('cos(θ)', fontsize=11)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('cos(θ) Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_single_distribution(self, angle_type: str = 'polar', 
                                 output_file: Optional[str] = None) -> None:
        """Plot single angle distribution (polar or azimuthal)"""
        
        if angle_type == 'polar':
            angles = [self.analyzer.polar_angle(p) for p in self.particles]
            angles_deg = np.degrees(angles)
            xlabel = 'Polar Angle θ (degrees)'
            title = 'Polar Angle Distribution'
            xlim = (0, 180)
            color = 'steelblue'
        elif angle_type == 'azimuthal':
            angles = [self.analyzer.azimuthal_angle(p) for p in self.particles]
            angles_deg = np.degrees(angles)
            xlabel = 'Azimuthal Angle φ (degrees)'
            title = 'Azimuthal Angle Distribution'
            xlim = (0, 360)
            color = 'darkgreen'
        else:  # pseudorapidity
            angles = [self.analyzer.pseudorapidity(p) for p in self.particles]
            angles_deg = np.array(angles)
            xlabel = 'Pseudorapidity η'
            title = 'Pseudorapidity Distribution'
            xlim = None
            color = 'purple'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(angles_deg, bins=40, color=color, edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Number of Particles', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        if xlim:
            ax.set_xlim(xlim)
        
        # Add statistics
        mean_angle = np.mean(angles_deg)
        std_angle = np.std(angles_deg)
        ax.text(0.98, 0.97, f'Mean: {mean_angle:.1f}\nStd: {std_angle:.1f}',
               transform=ax.transAxes, verticalalignment='top', 
               horizontalalignment='right', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=11)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_pt_distribution(self, output_file: Optional[str] = None) -> None:
        """Plot transverse momentum distribution"""
        pt = [self.analyzer.transverse_momentum(p) for p in self.particles]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(pt, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Transverse Momentum pT (GeV)', fontsize=12)
        ax.set_ylabel('Number of Particles', fontsize=12)
        ax.set_title('Transverse Momentum Distribution', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        # Add statistics
        mean_pt = np.mean(pt)
        std_pt = np.std(pt)
        ax.text(0.98, 0.97, f'Mean: {mean_pt:.2f} GeV\nStd: {std_pt:.2f} GeV',
               transform=ax.transAxes, verticalalignment='top', 
               horizontalalignment='right', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=11)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()