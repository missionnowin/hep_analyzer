from pathlib import Path
import sys
from typing import Iterator, List, override

try:
    from models.particle import Particle
    from utils.readers.reader import ReaderBase
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class PHQMDReader(ReaderBase):
    """PHQMD/UrQMD phsd.dat format reader."""
    
    @override
    def stream_batch(self, batch_size: int) -> Iterator[List[List[Particle]]]:
        """Stream events in batches."""
        batch = []
        
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Event header: typically starts with event number or time
            # Format varies, looking for line with multiple integers/floats
            if line and not line.startswith('#'):
                parts = line.split()
                
                # Try to parse as event header
                # Common format: event_num n_particles time ...
                if len(parts) >= 2:
                    try:
                        # Attempt to read as event
                        n_particles = int(parts[1]) if len(parts) >= 2 else 0
                        
                        if n_particles > 0 and n_particles < 10000:  # Sanity check
                            particles = []
                            
                            # Read particle lines
                            for j in range(n_particles):
                                if i + 1 + j < len(lines):
                                    p_line = lines[i + 1 + j].strip()
                                    if p_line and not p_line.startswith('#'):
                                        p_parts = p_line.split()
                                        
                                        # Parse particle: pid px py pz m [x y z t]
                                        if len(p_parts) >= 5:
                                            try:
                                                pid = int(p_parts[0])
                                                px = float(p_parts[1])
                                                py = float(p_parts[2])
                                                pz = float(p_parts[3])
                                                mass = float(p_parts[4])
                                                x = float(p_parts[5]) if len(p_parts) > 5 else 0.0
                                                y = float(p_parts[6]) if len(p_parts) > 6 else 0.0
                                                z = float(p_parts[7]) if len(p_parts) > 7 else 0.0
                                                t = float(p_parts[8]) if len(p_parts) > 8 else 0.0
                                                
                                                particles.append(Particle(pid, px, py, pz, mass, t, x, y, z))
                                            except (ValueError, IndexError):
                                                pass
                            
                            if particles:
                                batch.append(particles)
                                
                                if len(batch) >= batch_size:
                                    yield batch
                                    batch = []
                            
                            i += 1 + n_particles
                        else:
                            i += 1
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        if batch:
            yield batch