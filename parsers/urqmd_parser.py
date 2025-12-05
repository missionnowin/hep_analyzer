import re
import sys
from typing import Optional


try:
    from models.ions import CollisionSystem, IonDatabase
    from models.file_format import FileFormat
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class UrQMDParser:
    @staticmethod
    def parse_urqmd_file(file_path: str, fmt: FileFormat) -> Optional[CollisionSystem]:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if len(lines) < 3:
                return None

            header_line = lines[2].strip()

            return UrQMDParser._parse_urqmd_header(header_line)

        except Exception:
            return None
        

    @staticmethod
    def _parse_urqmd_header(header_line: str) -> Optional[CollisionSystem]:
        try:
            # Parse collision system (A,Z)+(A,Z)
            pattern = r'\((\d+)\s*,\s*(\d+)\)\s*\+\s*\((\d+)\s*,\s*(\d+)\)'
            match = re.search(pattern, header_line)

            if not match:
                print(f"No collision system found in header")
                return None

            A1, Z1, A2, Z2 = map(int, match.groups())

            after_collision = header_line[match.end():]

            # Look for scientific notation or regular float
            energy_pattern = r'[+-]?\d+\.?\d*[Ee][+-]?\d+|[+-]?\d+\.\d+'
            energy_matches = re.findall(energy_pattern, after_collision)

            energy = 0.0
            if energy_matches:
                try:
                    energy = float(energy_matches[0])
                except ValueError:
                    energy = 0.0

            projectile = IonDatabase.get_ion(Z1, A1)
            target = IonDatabase.get_ion(Z2, A2)

            return CollisionSystem(projectile, target, energy)

        except Exception:
            return None
