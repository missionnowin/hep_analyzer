from typing import Dict, List, Tuple

import numpy as np

from models.particle import Particle


class CMToLabTransformer:
    # Nucleon rest mass (GeV/c²)
    M_NUCLEON = 0.939565
    
    def __init__(self, gamma: float, beta: float):
        self.gamma = gamma
        self.beta = beta
    
    @classmethod
    def from_collision_energy(cls, sqrt_s_nn: float, A1: int = 197, A2: int = 197) -> 'CMToLabTransformer':
        m = cls.M_NUCLEON
        
        if A1 == A2:
            gamma = (sqrt_s_nn**2 + 2 * m**2) / (2 * m**2)
        else:
            gamma = (sqrt_s_nn**2 + 2 * m**2) / (2 * m**2)
        
        # Calculate β from γ
        beta = np.sqrt(1.0 - 1.0 / (gamma**2)) if gamma > 1.0 else 0.0
        
        return cls(gamma=gamma, beta=beta)
    
    @classmethod
    def from_system_info(cls, system_info: Dict) -> 'CMToLabTransformer':
        sqrt_s_nn = system_info.get('sqrt_s_NN', 200.0)
        A1 = system_info.get('A1', 197)
        A2 = system_info.get('A2', 197)
        
        return cls.from_collision_energy(sqrt_s_nn, A1=A1, A2=A2)
    
    def transform_momentum(
        self,
        px_cm: float,
        py_cm: float,
        pz_cm: float,
        energy_cm: float
    ) -> Tuple[float, float, float, float]:
        px_lab = px_cm
        py_lab = py_cm
        pz_lab = self.gamma * (pz_cm + self.beta * energy_cm)
        E_lab = self.gamma * (energy_cm + self.beta * pz_cm)
        
        return px_lab, py_lab, pz_lab, E_lab
    
    def transform_particle(self, particle: Particle):
        # Get CM values
        px_cm = particle.px
        py_cm = particle.py
        pz_cm = particle.pz
        energy_cm = particle.E
        
        # Transform momentum and energy
        px_lab, py_lab, pz_lab, E_lab = self.transform_momentum(
            px_cm, py_cm, pz_cm, energy_cm
        )
        
        # Update particle in-place
        particle.px = px_lab
        particle.py = py_lab
        particle.pz = pz_lab
        particle.E = E_lab
        
        return particle
    
    
    def transform_event(self, particles: List[Particle]) -> List[Particle]:
        for particle in particles:
            self.transform_particle(particle)
        return particles
    

    def transform_batch(self, batch: List[List[Particle]]) -> List[List[Particle]]:
        for event in batch:
            self.transform_event(event)
        return batch
    