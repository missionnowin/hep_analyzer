import sys
from typing import Dict


try:
    from models.ions import CollisionSystem
    from models.file_format import FileFormat
    from parsers.hepmc_parser import HepMCParser
    from parsers.phqmd_parser import PHQMDParser
    from parsers.urqmd_parser import UrQMDParser
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CollisionSystemDetector:
    @staticmethod
    def detect_from_file(file_path: str, 
                        file_format: str = None) -> Dict:
        """
        Detect collision parameters from file header.

        Returns:
            {
                'A1': mass of projectile (A number),
                'A2': mass of target (A number),
                'Z1': charge of projectile (atomic number),
                'Z2': charge of target (atomic number),
                'sqrt_s_NN': collision energy in GeV,
                'label': human-readable label (e.g., "Xe-129+Au-197 @ 3.0 GeV"),
                'confidence': confidence score (0-1),
            }

        Returns None if parsing fails.
        """
        try:
            if file_format:
                fmt = FileFormat[file_format.upper()]
            else:
                fmt = CollisionSystemDetector._detect_format(file_path)
            
            if fmt == FileFormat.UNKNOWN:
                return None
            
            # Route to appropriate parser
            parsers = {
                FileFormat.URQMD: UrQMDParser.parse_urqmd_file,
                FileFormat.PHQMD: PHQMDParser.parse_phqmd_file,
                FileFormat.HEPMC2: HepMCParser.parse_hepmc_file,
                FileFormat.HEPMC3: HepMCParser.parse_hepmc_file,
            }
            
            parser = parsers.get(fmt)
            
            if not parser:
                return None
            
            collision_system = parser(file_path, fmt)


            if collision_system:
                result = CollisionSystemDetector._collision_system_to_dict(collision_system)
                result['format'] = fmt.value
                return result
            
        except Exception:
            return None
        
        return None

    @staticmethod
    def _detect_format(file_path: str) -> FileFormat:
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4096)  # Read first 4KB
            
            # Check HepMC3 first (most specific)
            if b'HepMC3' in header or b'# PYTHIA EVENT FILE' in header:
                return FileFormat.HEPMC3
            
            # Check HepMC2
            if b'HepMC::' in header or (b'E ' in header and b'V ' in header):
                return FileFormat.HEPMC2
            
            # Check PHQMD
            if b'PHQMD' in header:
                return FileFormat.PHQMD
            
            # Check UrQMD (most common, check last)
            if b'UrQMD' in header or b'OSCAR' in header or b'final_id_p_x' in header:
                return FileFormat.URQMD
            
            return FileFormat.UNKNOWN
            
        except Exception as e:
            print(f"Error detecting format: {e}")
            return FileFormat.UNKNOWN

    @staticmethod
    def _collision_system_to_dict(sys: CollisionSystem) -> Dict:
        return {
            'A1': sys.projectile.A,
            'A2': sys.target.A,
            'Z1': sys.projectile.Z,
            'Z2': sys.target.Z,
            'sqrt_s_NN': sys.collision_energy,
            'label': sys.label,
            'projectile_name': sys.projectile.name,
            'target_name': sys.target.name,
            'confidence': 1.0,
        }

   