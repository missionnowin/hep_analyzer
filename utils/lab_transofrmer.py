from typing import Dict, List, Tuple

import numpy as np

from models.particle import Particle


class CMToLabTransformer:
    # Nucleon rest mass (GeV/c²)
    
    def __init__(self, gamma: float, beta: float):
        self.gamma = gamma
        self.beta = beta

    
    @classmethod
    def from_collision_energy(cls, e_kin_lab: float, A1: int, A2: int) -> 'CMToLabTransformer':   
        # Nucleon mass (GeV)
        M_NUCLEON = 0.938
        # Calculate masses
        M1 = A1 * M_NUCLEON  # projectile mass
        M2 = A2 * M_NUCLEON  # target mass
        E_kin_lab = e_kin_lab * M1

        beta_cm = np.sqrt(E_kin_lab**2 + 2 * E_kin_lab * M1) / (E_kin_lab + M1 + M2)
        gamma_cm = 1 / np.sqrt(1 - beta_cm**2)

        return cls(gamma=gamma_cm, beta=beta_cm)

    
    
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
        pz_lab = self.gamma * (-pz_cm + self.beta * energy_cm)
        E_lab = self.gamma * (energy_cm - self.beta * pz_cm)
        
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
    