from pathlib import Path
import sys
from typing import List, Optional, Union, Generator, override
from enum import Enum

try:
    from models.particle import Particle
    from readers.reader import ReaderBase
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class OscarFormat(Enum):
    """Supported Oscar format types"""
    OSCAR1992A = "oscar1992a"
    OSC1997A = "osc1997a"


class OscarFormatDetector:
    """Automatically detect Oscar format type from file"""

    @staticmethod
    def detect_format(filepath: str) -> OscarFormat:
        """Read first few lines and determine format"""
        with open(filepath, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(3)]
        
        if any('OSC1997A' in line or 'final_id_p_x' in line for line in first_lines):
            return OscarFormat.OSC1997A
        if any('OSCAR' in line.upper() for line in first_lines):
            return OscarFormat.OSCAR1992A
        
        return OscarFormat.OSCAR1992A


class Oscar1992AReader:
    """Read and parse Oscar1992A format files"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
    
    def stream_batch(self, batch_size: int) -> Generator[List[List[Particle]], None, None]:
        """Parse Oscar1992A file and return list of events with particles"""
        with open(self.filepath, 'r') as f:
            batch = []
            
            while True:
                header_line = f.readline().strip()
                if not header_line:
                    if batch:
                        yield batch
                    break
                
                try:
                    header = header_line.split()
                    if len(header) < 2:
                        continue
                    
                    n_particles = int(header[1])
                    particles = []
                    
                    for _ in range(n_particles):
                        particle_line = f.readline().strip()
                        if not particle_line:
                            break
                        
                        p_data = particle_line.split()
                        if len(p_data) < 12:
                            continue
                        
                        try:
                            particle = Particle(
                                particle_id=int(p_data[0]),
                                mass=float(p_data[1]),
                                x=float(p_data[2]),
                                y=float(p_data[3]),
                                z=float(p_data[4]),
                                px=float(p_data[5]),
                                py=float(p_data[6]),
                                pz=float(p_data[7]),
                                E=float(p_data[8]),
                                t_formed=float(p_data[9]),
                                charge=int(p_data[10]),
                                baryon_density=float(p_data[11])
                            )
                            particles.append(particle)
                        except (ValueError, IndexError):
                            continue
                    
                    if particles:
                        batch.append(particles)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                
                except (ValueError, IndexError):
                    continue


class OSC1997AReader:
    """Read and parse OSC1997A (final_id_p_x) format files"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
    
    def stream_batch(self, batch_size: int) -> Generator[List[List[Particle]], None, None]:
        """Parse OSC1997A file and return list of events with particles"""
        with open(self.filepath, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            
            batch = []
            
            while True:
                header_line = f.readline().strip()
                if not header_line:
                    if batch:
                        yield batch
                    break
                
                header = header_line.split()
                if len(header) < 2:
                    continue
                
                try:
                    n_particles = int(header[1])
                except ValueError:
                    continue
                
                particles = []
                for _ in range(n_particles):
                    particle_line = f.readline().strip()
                    if not particle_line:
                        break
                    
                    cols = particle_line.split()
                    if len(cols) < 11:
                        continue
                    
                    try:
                        particle = Particle(
                            particle_id=int(cols[1]),
                            mass=float(cols[6]),
                            x=float(cols[7]),
                            y=float(cols[8]),
                            z=float(cols[9]),
                            px=float(cols[2]),
                            py=float(cols[3]),
                            pz=float(cols[4]),
                            E=float(cols[5]),
                            t_formed=float(cols[10]),
                            charge=None,
                            baryon_density=None
                        )
                        particles.append(particle)
                    except (ValueError, IndexError):
                        continue
                
                if particles:
                    batch.append(particles)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []


class OscarReader(ReaderBase):
    """Universal Oscar reader - auto-detects format and parses accordingly"""
    @override
    def __init__(self, filepath: str, format_type: Optional[OscarFormat] = None):
        self.filepath = Path(filepath)
        self.format_type = format_type or OscarFormatDetector.detect_format(str(filepath))
        self._reader = self._create_reader()
    
    def _create_reader(self) -> Union[Oscar1992AReader, OSC1997AReader]:
        if self.format_type == OscarFormat.OSCAR1992A:
            return Oscar1992AReader(str(self.filepath))
        else:
            return OSC1997AReader(str(self.filepath))
    
    @override
    def stream_batch(self, batch_size: int) -> Generator[List[List[Particle]], None, None]:
        return self._reader.stream_batch(batch_size)