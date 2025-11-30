import numpy as np
from typing import Dict, Tuple


class CollisionSystemDetector:
    """
    Automatically infer collision system (A1, A2, Z1, Z2, sqrt(s_NN))
    from Oscar event data.
    """
    
    @staticmethod
    def detect_from_file(file_path: str, n_events_sample: int = 100) -> Dict:
        """
        Analyze first N events to infer collision parameters.
        
        Returns:
            {
                'A1': mass of projectile,
                'A2': mass of target,
                'Z1': charge of projectile,
                'Z2': charge of target,
                'sqrt_s_NN': estimated collision energy (GeV),
                'label': human-readable label (e.g., "Au+Au @ 10 GeV"),
                'confidence': confidence score (0-1)
            }
        """
        try:
            from utils.readers import OscarReader
        except ImportError:
            print("Error: Cannot import OscarReader")
            return None
        
        reader = OscarReader(file_path)
        
        # Sample statistics
        total_particles = 0
        charge_counts = {}  # Track charge distribution
        pdgid_counts = {}   # Track particle types
        max_pz_forward = []
        max_pz_backward = []
        pz_means = []
        
        event_count = 0
        for particles in reader.read_file():
            if event_count >= n_events_sample:
                break
            
            if not particles:
                continue
            
            total_particles += len(particles)
            
            # Count charges (nucleons: protons Z=1, neutrons Z=0)
            for p in particles:
                pdgid = p.particle_id
                charge = p.charge
                
                # Track protons (Z=1) and neutrons (Z=0) for nuclei
                if pdgid in [2212, 2112]:  # proton, neutron
                    charge_counts[charge] = charge_counts.get(charge, 0) + 1
                
                pdgid_counts[pdgid] = pdgid_counts.get(pdgid, 0) + 1
            
            # Track pz distribution (hints at collision energy)
            pz_vals = [p.pz for p in particles]
            if pz_vals:
                max_pz_forward.append(max([p for p in pz_vals if p > 0], default=0))
                max_pz_backward.append(min([p for p in pz_vals if p < 0], default=0))
                pz_means.append(np.mean(pz_vals))
            
            event_count += 1
        
        # Infer nuclei
        A1, Z1, A2, Z2 = CollisionSystemDetector._infer_nuclei(
            charge_counts, pdgid_counts, total_particles, event_count
        )
        
        # Infer energy from max pz
        sqrt_s_NN = CollisionSystemDetector._infer_energy(
            max_pz_forward, max_pz_backward, pz_means
        )
        
        # Generate label
        label = f"{CollisionSystemDetector._nucleus_name(Z1, A1)}+"
        label += f"{CollisionSystemDetector._nucleus_name(Z2, A2)} @ {sqrt_s_NN:.1f} GeV"
        
        # Confidence score based on data consistency
        confidence = CollisionSystemDetector._assess_confidence(
            charge_counts, pdgid_counts, total_particles, event_count
        )
        
        return {
            'A1': A1,
            'A2': A2,
            'Z1': Z1,
            'Z2': Z2,
            'sqrt_s_NN': sqrt_s_NN,
            'label': label,
            'confidence': confidence,
            'n_events_analyzed': event_count,
            'n_particles_total': total_particles,
            'avg_particles_per_event': total_particles / event_count if event_count > 0 else 0
        }
    
    @staticmethod
    def _infer_nuclei(charge_counts: Dict, pdgid_counts: Dict, 
                      total_particles: int, n_events: int) -> Tuple[int, int, int, int]:
        """
        Infer nucleus identity from charge distribution.
        
        Returns: (A1, Z1, A2, Z2)
        """
        # Common heavy-ion systems
        NUCLEI = {
            'Au': (197, 79),   # Gold
            'Pb': (207, 82),   # Lead
            'U': (238, 92),    # Uranium
            'Cu': (63, 29),    # Copper
            'Ni': (58, 28),    # Nickel
            'd': (2, 1),       # Deuteron
            'p': (1, 1),       # Proton
            'n': (1, 0),       # Neutron
        }
        
        # Average particles per event
        avg_per_event = total_particles / n_events if n_events > 0 else 0
        
        # Try to match based on multiplicity
        best_match = ('Au', 'Au', 197, 79, 197, 79)
        
        if avg_per_event > 500:
            best_match = ('Au', 'Au', 197, 79, 197, 79)  # Au+Au
        elif avg_per_event > 200:
            best_match = ('Pb', 'Pb', 207, 82, 207, 82)  # Pb+Pb
        elif avg_per_event > 100:
            best_match = ('Cu', 'Cu', 63, 29, 63, 29)    # Cu+Cu
        elif avg_per_event > 50:
            best_match = ('p', 'Au', 1, 1, 197, 79)      # p+Au
        
        return best_match[2], best_match[3], best_match[4], best_match[5]
    
    @staticmethod
    def _infer_energy(max_pz_forward: list, max_pz_backward: list, 
                     pz_means: list) -> float:
        """
        Infer collision energy from pz distribution.
        
        Returns: sqrt(s_NN) in GeV
        """
        if not max_pz_forward or not max_pz_backward:
            return 10.0  # Default fallback
        
        avg_max_pz_fwd = np.mean(max_pz_forward)
        avg_max_pz_bwd = np.mean([abs(p) for p in max_pz_backward if p < 0])
        
        # Rough scaling: higher pz → higher energy
        # At 10 GeV: max_pz ~ 5 GeV
        # At 20 GeV: max_pz ~ 10 GeV
        # At 200 GeV: max_pz ~ 100 GeV
        
        estimated_max_pz = max(avg_max_pz_fwd, avg_max_pz_bwd)
        
        # Energy scales roughly as pz^2 for ultra-relativistic particles
        if estimated_max_pz < 2:
            sqrt_s_NN = 5.0
        elif estimated_max_pz < 5:
            sqrt_s_NN = 10.0
        elif estimated_max_pz < 10:
            sqrt_s_NN = 20.0
        elif estimated_max_pz < 50:
            sqrt_s_NN = 50.0
        elif estimated_max_pz < 100:
            sqrt_s_NN = 200.0
        else:
            sqrt_s_NN = 2760.0
        
        return sqrt_s_NN
    
    @staticmethod
    def _nucleus_name(Z: int, A: int) -> str:
        """Convert Z, A to nucleus name."""
        names = {
            (1, 1): 'p',
            (0, 1): 'n',
            (1, 2): 'd',
            (2, 3): '³He',
            (2, 4): '⁴He',
            (6, 12): '¹²C',
            (8, 16): '¹⁶O',
            (29, 63): 'Cu',
            (79, 197): 'Au',
            (82, 207): 'Pb',
            (92, 238): 'U',
        }
        
        if (Z, A) in names:
            return names[(Z, A)]
        else:
            # Fallback: use element symbol if available
            elements = {
                1: 'H', 2: 'He', 6: 'C', 8: 'O', 29: 'Cu', 79: 'Au', 82: 'Pb', 92: 'U'
            }
            elem = elements.get(Z, f'Z{Z}')
            return f'{elem}({A})'
    
    @staticmethod
    def _assess_confidence(charge_counts: Dict, pdgid_counts: Dict,
                          total_particles: int, n_events: int) -> float:
        """
        Assess confidence in system detection (0-1).
        Higher if data is clean and consistent.
        """
        confidence = 0.5  # Base confidence
        
        # More events → higher confidence
        if n_events >= 100:
            confidence += 0.2
        elif n_events >= 50:
            confidence += 0.1
        
        # More particles → higher confidence in statistics
        if total_particles > 10000:
            confidence += 0.2
        elif total_particles > 5000:
            confidence += 0.1
        
        # Consistency in particle types
        if len(pdgid_counts) < 20:  # Mostly one or two particle types
            confidence += 0.1
        
        return min(confidence, 1.0)


# Integration example
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python collision_system_detector.py <path_to_file.f19>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"\nAnalyzing: {file_path}")
    print("=" * 80)
    
    result = CollisionSystemDetector.detect_from_file(file_path, n_events_sample=100)
    
    if result:
        print(f"\nDetected Collision System:")
        print(f"  Label: {result['label']}")
        print(f"  Projectile: {result['_nucleus_name'](result['Z1'], result['A1'])} (Z={result['Z1']}, A={result['A1']})")
        print(f"  Target: {result['_nucleus_name'](result['Z2'], result['A2'])} (Z={result['Z2']}, A={result['A2']})")
        print(f"  Energy: sqrt(s_NN) = {result['sqrt_s_NN']:.1f} GeV")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Events analyzed: {result['n_events_analyzed']}")
        print(f"  Avg particles/event: {result['avg_particles_per_event']:.0f}")
