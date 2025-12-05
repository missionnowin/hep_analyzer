import re
import sys
from typing import Optional, Tuple


try:
    from models.ions import CollisionSystem
    from models.ions import CollisionSystem, IonDatabase
    from models.file_format import FileFormat
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class PHQMDParser:  
    @staticmethod
    def parse_phqmd_file(file_path: str, fmt: FileFormat) -> Optional[CollisionSystem]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 50 lines for metadata
                lines = [f.readline() for _ in range(50)]
            
            # Extract projectile, target, and energy
            projectile_info = PHQMDParser._extract_phqmd_projectile(lines)
            target_info = PHQMDParser._extract_phqmd_target(lines)
            energy = PHQMDParser._extract_phqmd_energy(lines)
            
            if not projectile_info or not target_info:
                print("Could not extract projectile/target from PHQMD file")
                return None
            
            Z1, A1 = projectile_info
            Z2, A2 = target_info
            
            projectile = IonDatabase.get_ion(Z1, A1)
            target = IonDatabase.get_ion(Z2, A2)
            
            return CollisionSystem(projectile, target, energy)
        
        except Exception as e:
            print(f"Error reading PHQMD file: {e}")
            return None
    
    @staticmethod
    def _extract_phqmd_projectile(lines: list) -> Optional[Tuple[int, int]]:
        combined_text = '\n'.join(lines).lower()
        
        # Pattern 1: "projectile = Au-197" or "projectile= au 197"
        match = re.search(r'projectile\s*=\s*([a-z]+)[-\s]?(\d+)', combined_text)
        if match:
            return PHQMDParser._parse_ion_spec(match.group(1), match.group(2))
        
        # Pattern 2: First ion in "Au-197 + Au-197 @ X GeV"
        match = re.search(r'([a-z]+)[-\s]?(\d+)\s*\+', combined_text)
        if match:
            return PHQMDParser._parse_ion_spec(match.group(1), match.group(2))
        
        return None
    
    @staticmethod
    def _extract_phqmd_target(lines: list) -> Optional[Tuple[int, int]]:
        combined_text = '\n'.join(lines).lower()
        
        # Pattern 1: "target = Au-197"
        match = re.search(r'target\s*=\s*([a-z]+)[-\s]?(\d+)', combined_text)
        if match:
            return PHQMDParser._parse_ion_spec(match.group(1), match.group(2))
        
        # Pattern 2: Second ion in "Au-197 + Au-197 @ X GeV"
        match = re.search(r'\+\s*([a-z]+)[-\s]?(\d+)\s*@', combined_text)
        if match:
            return PHQMDParser._parse_ion_spec(match.group(1), match.group(2))
        
        return None
    
    @staticmethod
    def _extract_phqmd_energy(lines: list) -> float:
        combined_text = '\n'.join(lines).lower()
        
        # Pattern 1: "energy = X GeV" or "energy= X gev"
        match = re.search(r'energy\s*=\s*([+-]?\d+\.?\d*[Ee]?[+-]?\d*)\s*gev', combined_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Pattern 2: "@ X GeV"
        match = re.search(r'@\s*([+-]?\d+\.?\d*[Ee]?[+-]?\d*)\s*gev', combined_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Pattern 3: "sqrt_s_NN = X GeV"
        match = re.search(r'sqrt_s_nn\s*=\s*([+-]?\d+\.?\d*[Ee]?[+-]?\d*)\s*gev', combined_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return 0.0
    
    @staticmethod
    def _parse_ion_spec(element: str, mass_str: str) -> Optional[Tuple[int, int]]:
        try:
            A = int(mass_str)
        except ValueError:
            return None
        
        Z = IonDatabase.element_name_to_z(element)
        
        if Z is None:
            return None
        
        return (Z, A)