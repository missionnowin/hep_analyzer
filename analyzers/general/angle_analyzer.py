import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

try:
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