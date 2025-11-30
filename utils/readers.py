from pathlib import Path
import sys
from typing import List, Optional, Union
from enum import Enum


try:
    from models.particle import Particle
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
        
        # Check for OSC1997A markers
        if any('OSC1997A' in line or 'final_id_p_x' in line for line in first_lines):
            return OscarFormat.OSC1997A
        
        # Check for Oscar1992A marker or pattern
        if any('OSCAR' in line.upper() for line in first_lines):
            return OscarFormat.OSCAR1992A
        
        # Default heuristic: Oscar1992A
        return OscarFormat.OSCAR1992A


class Oscar1992AReader:
    """Read and parse Oscar1992A format files"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.events = []
        
    def read_file(self) -> List[List[Particle]]:
        """Parse Oscar1992A file and return list of events with particles"""
        events = []
        
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Parse event header
            try:
                header = line.split()
                if len(header) < 2:
                    i += 1
                    continue
                
                t = float(header[0])
                n_particles = int(header[1])
                
                # Read particle data
                particles = []
                for j in range(n_particles):
                    i += 1
                    if i >= len(lines):
                        break
                    
                    particle_line = lines[i].strip()
                    if particle_line:
                        p_data = particle_line.split()
                        if len(p_data) < 12:
                            continue
                        
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
                
                if particles:
                    events.append(particles)
                    
            except (ValueError, IndexError):
                pass
            
            i += 1
        
        self.events = events
        return events


class OSC1997AReader:
    """Read and parse OSC1997A (final_id_p_x) format files"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.events = []
        
    def read_file(self) -> List[List[Particle]]:
        """Parse OSC1997A file and return list of events with particles"""
        events = []
        
        with open(self.filepath, 'r') as f:
            lines = [l.rstrip('\n') for l in f if l.strip()]
        
        # Skip global header lines (OSC1997A, final_id_p_x, UrQMD info)
        i = 3 if len(lines) > 3 else 0
        
        while i < len(lines):
            header = lines[i].split()
            
            # Expect: event_id n_particles b phi
            if len(header) < 2:
                i += 1
                continue
            
            try:
                event_id = int(header[0])
                n_particles = int(header[1])
            except ValueError:
                i += 1
                continue
            
            # Collect particle lines
            particles = []
            for j in range(n_particles):
                i += 1
                if i >= len(lines):
                    break
                
                cols = lines[i].split()
                if len(cols) < 11:
                    continue
                
                try:
                    # OSC1997A format:
                    # index  pdg  px  py  pz  E  m  x  y  z  t
                    idx = int(cols[0])
                    pdg = int(cols[1])
                    px = float(cols[2])
                    py = float(cols[3])
                    pz = float(cols[4])
                    E = float(cols[5])
                    m = float(cols[6])
                    x = float(cols[7])
                    y = float(cols[8])
                    z = float(cols[9])
                    t = float(cols[10])
                    
                    particle = Particle(
                        particle_id=pdg,
                        mass=m,
                        x=x,
                        y=y,
                        z=z,
                        px=px,
                        py=py,
                        pz=pz,
                        E=E,
                        t_formed=t,
                        charge=None,
                        baryon_density=None
                    )
                    particles.append(particle)
                    
                except (ValueError, IndexError):
                    continue
            
            if particles:
                events.append(particles)
            
            i += 1
        
        self.events = events
        return events


class OscarReader:
    """Universal Oscar reader - auto-detects format and parses accordingly"""
    
    def __init__(self, filepath: str, format_type: Optional[OscarFormat] = None):
        self.filepath = Path(filepath)
        self.format_type = format_type or OscarFormatDetector.detect_format(str(filepath))
        self.events = []
        self._reader = self._create_reader()
        
    def _create_reader(self) -> Union[Oscar1992AReader, OSC1997AReader]:
        """Create appropriate reader based on format"""
        if self.format_type == OscarFormat.OSCAR1992A:
            return Oscar1992AReader(str(self.filepath))
        else:  # OSC1997A
            return OSC1997AReader(str(self.filepath))
    
    def read_file(self) -> List[List[Particle]]:
        """Parse file and return list of events"""
        self.events = self._reader.read_file()
        return self.events