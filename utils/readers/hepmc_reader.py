import sys
from typing import Iterator, List, override

try:
    from models.particle import Particle
    from utils.readers.reader import ReaderBase
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class HepMCReader(ReaderBase):
    """HepMC ASCII format reader (Pythia/Herwig generator output)."""
    
    @override
    def stream_batch(self, batch_size: int) -> Iterator[List[List[Particle]]]:
        """Stream events in batches."""
        batch = []
        
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Event line: "E event_number N_vertices N_particles"
            if line.startswith('E '):
                particles = []
                i += 1
                
                # Read vertex/particle lines until next event or EOF
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Vertex line: "V id status N_parents N_children X Y Z ctau"
                    if line.startswith('V '):
                        i += 1
                        continue
                    
                    # Particle line: "P id mother1 mother2 color1 color2 px py pz E m lifetime status"
                    elif line.startswith('P '):
                        parts = line.split()
                        if len(parts) >= 11:
                            try:
                                pid = int(parts[1])
                                px = float(parts[6])
                                py = float(parts[7])
                                pz = float(parts[8])
                                energy = float(parts[9])
                                mass = float(parts[10])
                                status = int(parts[11]) if len(parts) > 11 else 1
                                
                                # Only include final state particles (status=1)
                                if status == 1:
                                    particles.append(Particle(pid, px, py, pz, mass))
                            except (ValueError, IndexError):
                                pass
                        i += 1
                    
                    # End event marker
                    elif line.startswith('E ') or not line or line.startswith('#'):
                        break
                    else:
                        i += 1
                
                if particles:
                    batch.append(particles)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            else:
                i += 1
        
        if batch:
            yield batch