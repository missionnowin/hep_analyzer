import sys
from abc import ABC, abstractmethod
from typing import Iterator, List

try:
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class ReaderBase(ABC):
    """Abstract base class for all event data readers."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def stream_batch(self, batch_size: int) -> Iterator[List[List[Particle]]]:
        pass
