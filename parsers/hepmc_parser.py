import sys
from typing import Optional, Tuple


try:
    from models.file_format import FileFormat
    from models.ions import CollisionSystem, IonDatabase
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class HepMCParser:
    @staticmethod
    def parse_hepmc_file(file_path: str, fmt: FileFormat) -> Optional[CollisionSystem]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if fmt == FileFormat.HEPMC2:
                return HepMCParser._parse_hepmc2_lines(lines)
            else:  # HEPMC3
                return HepMCParser._parse_hepmc3_lines(lines)
        
        except Exception as e:
            print(f"Error reading HepMC file: {e}")
            return None
    
    @staticmethod
    def _parse_hepmc2_lines(lines: list) -> Optional[CollisionSystem]:
        try:
            projectile_pdg = None
            target_pdg = None
            energy = 0.0
            
            for i, line in enumerate(lines[:200]):  # Check first 200 lines
                # Parse event line: E  event_number ...
                if line.startswith('E '):
                    parts = line.split()
                    if len(parts) >= 13:
                        try:
                            energy = float(parts[12])  # sqrt_s
                        except (ValueError, IndexError):
                            pass
                
                # Parse particle lines: P id ... momentum ...
                if line.startswith('P '):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            pdg_id = int(parts[1])
                            status = int(parts[2])
                            
                            # Status 3 or 4 = incoming particles
                            if status in [3, 4]:
                                if projectile_pdg is None:
                                    projectile_pdg = pdg_id
                                elif target_pdg is None:
                                    target_pdg = pdg_id
                        except ValueError:
                            pass
            
            if projectile_pdg is None or target_pdg is None:
                print("Could not extract incoming particles from HepMC2 file")
                return None
            
            # Convert PDG IDs to (Z, A)
            proj_za = HepMCParser._pdg_to_za(projectile_pdg)
            targ_za = HepMCParser._pdg_to_za(target_pdg)
            
            if not proj_za or not targ_za:
                return None
            
            Z1, A1 = proj_za
            Z2, A2 = targ_za
            
            projectile = IonDatabase.get_ion(Z1, A1)
            target = IonDatabase.get_ion(Z2, A2)
            
            return CollisionSystem(projectile, target, energy)
        
        except Exception:
            return None
    
    @staticmethod
    def _parse_hepmc3_lines(lines: list) -> Optional[CollisionSystem]:
        try:
            projectile_pdg = None
            target_pdg = None
            energy = 0.0
            
            for i, line in enumerate(lines[:300]):
                if line.startswith('E '):
                    parts = line.split()
                    if len(parts) >= 13:
                        try:
                            energy = float(parts[12])
                        except (ValueError, IndexError):
                            pass
                
                if line.startswith('P '):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            pdg_id = int(parts[2])
                            status_str = parts[3] if len(parts) > 3 else '0'
                            status = int(status_str)
                            
                            # Status 4 = incoming in HepMC3
                            if status == 4:
                                if projectile_pdg is None:
                                    projectile_pdg = pdg_id
                                elif target_pdg is None:
                                    target_pdg = pdg_id
                        except ValueError:
                            pass
            
            if projectile_pdg is None or target_pdg is None:
                print("Could not extract incoming particles from HepMC3 file")
                return None
            
            proj_za = HepMCParser._pdg_to_za(projectile_pdg)
            targ_za = HepMCParser._pdg_to_za(target_pdg)
            
            if not proj_za or not targ_za:
                return None
            
            Z1, A1 = proj_za
            Z2, A2 = targ_za
            
            projectile = IonDatabase.get_ion(Z1, A1)
            target = IonDatabase.get_ion(Z2, A2)
            
            return CollisionSystem(projectile, target, energy)
        
        except Exception:
            return None
    
    @staticmethod
    def _pdg_to_za(pdg_id: int) -> Optional[Tuple[int, int]]:
        pdg_id = int(pdg_id)
        
        # Handle single nucleons
        if pdg_id == 2212:  # Proton
            return (1, 1)
        elif pdg_id == 2112:  # Neutron
            return (0, 1)
        
        # Extract from standard nucleus format: 10LZZZAAAI
        if pdg_id > 1000000000:
            # Remove leading 10 and trailing isomer flag
            nucleus_code = pdg_id - 1000000000
            
            # ZZZ is digits 3-5 from left (000zzz000)
            Z = (nucleus_code // 10000) % 1000
            # AAA is digits 6-8 from left (000000aaa)
            A = nucleus_code % 10000
            
            if Z > 0 and A >= Z:
                return (Z, A)
        
        return None

