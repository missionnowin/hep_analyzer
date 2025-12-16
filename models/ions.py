from dataclasses import dataclass
from typing import Dict, Optional

MUONS = [
    13,
    -13
]

PIONS = [
    211,
    -211,
]

PROTONS = [
    2212,
]

CHARGED_PDGIDS = [
    *MUONS,
    *PIONS,
    *PROTONS
]


@dataclass
class Ion:
    """Represents a nucleus with Z (charge) and A (mass number)."""
    Z: int  # Atomic number (protons)
    A: int  # Mass number (nucleons)

    @property
    def name(self) -> str:
        """Return human-readable nucleus name."""
        element_names = {
            0: 'n',    1: 'p/H',  2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',
            8: 'O',    9: 'F',   10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
           16: 'S',   17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',
           24: 'Cr',  25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
           32: 'Ge',  33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',
           40: 'Zr',  41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
           48: 'Cd',  49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs',
           56: 'Ba',  57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu',
           64: 'Gd',  65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
           72: 'Hf',  73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
           80: 'Hg',  81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr',
           88: 'Ra',  89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am',
           96: 'Cm',
        }
        elem = element_names.get(self.Z, f'Z{self.Z}')

        # Special cases
        if self.Z == 0 and self.A == 1:
            return 'n'
        if self.Z == 1 and self.A == 1:
            return 'p'
        if self.Z == 1 and self.A == 2:
            return 'd'
        if self.Z == 1 and self.A == 3:
            return 't'

        return f'{elem}-{self.A}' if self.A > 0 else elem

    def __repr__(self) -> str:
        return f'Ion(Z={self.Z}, A={self.A}, name={self.name!r})'


class IonDatabase:
    """Complete database of ions for quick lookup."""

    # Pre-defined commonly used ions
    IONS: Dict[str, Ion] = {
        'neutron': Ion(0, 1),
        'proton': Ion(1, 1),
        'deuteron': Ion(1, 2),
        'triton': Ion(1, 3),
        'C12': Ion(6, 12),
        'N14': Ion(7, 14),
        'O16': Ion(8, 16),
        'Cu63': Ion(29, 63),
        'Cu64': Ion(29, 64),
        'Cu65': Ion(29, 65),
        'Xe129': Ion(54, 129),
        'Au197': Ion(79, 197),
        'Pb207': Ion(82, 207),
        'U238': Ion(92, 238),
    }

    # Build reverse element name -> Z mapping from Ion.name() element_names dict
    _ELEMENT_NAME_TO_Z: Optional[Dict[str, int]] = None

    @classmethod
    def _build_element_map(cls) -> Dict[str, int]:
        """
        Build element symbol to Z mapping from Ion.name element_names.
        
        Handles all variants (e.g., 'p/H' → 'p', 'h')
        Cached after first call.
        
        Returns:
            Dictionary mapping lowercase element symbols to atomic numbers
        """
        if cls._ELEMENT_NAME_TO_Z is not None:
            return cls._ELEMENT_NAME_TO_Z

        element_names = {
            0: 'n',    1: 'p/H',  2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',
            8: 'O',    9: 'F',   10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
           16: 'S',   17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',
           24: 'Cr',  25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
           32: 'Ge',  33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',
           40: 'Zr',  41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
           48: 'Cd',  49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs',
           56: 'Ba',  57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu',
           64: 'Gd',  65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
           72: 'Hf',  73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
           80: 'Hg',  81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr',
           88: 'Ra',  89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am',
           96: 'Cm',
        }

        element_to_z = {}
        for z, elem in element_names.items():
            # Handle multi-variant entries like 'p/H' → both 'p' and 'h'
            for variant in elem.split('/'):
                element_to_z[variant.lower()] = z

        cls._ELEMENT_NAME_TO_Z = element_to_z
        return element_to_z

    @classmethod
    def element_name_to_z(cls, element: str) -> Optional[int]:
        """
        Convert element symbol to atomic number.
        
        Args:
            element: Element symbol (e.g., 'au', 'xe', 'pb')
        
        Returns:
            Atomic number (Z) or None if unknown
        """
        element_map = cls._build_element_map()
        element_clean = element.strip().lower()
        return element_map.get(element_clean)

    @classmethod
    def get_ion(cls, Z: int, A: int) -> Ion:
        """Get ion by Z and A."""
        return Ion(Z, A)

    @classmethod
    def get_ion_by_name(cls, name: str) -> Optional[Ion]:
        """Get ion by common name (e.g., 'Au197', 'proton')."""
        return cls.IONS.get(name.lower())

    @classmethod
    def parse_ion(cls, Z: int, A: int) -> Ion:
        """Parse ion from Z and A values."""
        return cls.get_ion(Z, A)


@dataclass
class CollisionSystem:
    """Represents a collision system with projectile and target."""
    projectile: Ion
    target: Ion
    collision_energy: float  # sqrt(s_NN) in GeV
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = f"{self.projectile.name}+{self.target.name} @ {self.collision_energy:.1f} GeV"

    def __repr__(self) -> str:
        return f'CollisionSystem({self.label})'