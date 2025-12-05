from pathlib import Path
import sys
from typing import Optional, Tuple


try:
    from readers.oscar_reader import OscarReader
    from readers.hepmc_reader import HepMCReader
    from readers.phqmd_reader import PHQMDReader
    from readers.reader import ReaderBase
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class MultiFormatReader:
    """Universal reader with auto-detection."""
    
    FORMAT_DETECTORS = {
        '.f19': ('oscar', OscarReader),
        '.hepmc': ('hemc', HepMCReader),
        '.dat': ('phqmd', PHQMDReader),
    }

    FORMAT_MAP = {
        'oscar': OscarReader,
        'hepmc': HepMCReader,
        'phqmd': PHQMDReader,
    }
    
    @staticmethod
    def detect_format(filepath: str, format_override: Optional[str] = None) -> Tuple[str, type]:
        """Auto-detect file format from extension and header."""
        path = Path(filepath)
        # Check by extension
        for ext, (fmt_name, reader_class) in MultiFormatReader.FORMAT_DETECTORS.items():
            if str(path).endswith(ext):
                return fmt_name, reader_class
        
        # Try header inspection if extension unknown
        try:
            with open(filepath, 'r') as f:
                header = f.read(500)
            
            if header.startswith('E '):
                return 'hepmc', HepMCReader
            elif 'V ' in header[:100] or 'P ' in header[:100]:
                return 'hepmc', HepMCReader
            else:
                # Default to OSCAR
                return 'oscar', OscarReader
        except:
            return 'oscar', OscarReader
    
    @staticmethod
    def open(filepath: str, format_override: Optional[str] = None) -> ReaderBase:
        """Open file with auto-detected format or user-specified format.
        
        Args:
            filepath: Path to file
            format_override: Optional format to force ('oscar', 'hepmc', 'phqmd')
                            If None, auto-detects from file
        
        Returns:
            Reader instance (OscarReader, HepMCReader, or PHQMDReader)
        
        Raises:
            ValueError: If format_override is invalid
        """
        if format_override:
            if format_override.lower() not in MultiFormatReader.FORMAT_MAP:
                raise ValueError(
                    f"Invalid format: {format_override}. "
                    f"Must be one of: {list(MultiFormatReader.FORMAT_MAP.keys())}"
                )
            reader_class = MultiFormatReader.FORMAT_MAP[format_override.lower()]
            return reader_class(filepath)
        else:
            fmt_name, reader_class = MultiFormatReader.detect_format(filepath, format_override)
        return reader_class(filepath)