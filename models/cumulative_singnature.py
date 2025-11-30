from dataclasses import dataclass, field


@dataclass
class CumulativeSignature:
    signature_type: str
    strength: float                    # 0-1, magnitude of effect
    confidence: float                  # 0-1, statistical confidence
    affected_particles: int            # Number of particles affected
    description: str = field(default="")
    
    def __post_init__(self):
        """Auto-generate description if not provided"""
        if not self.description:
            self.description = (
                f"{self.signature_type}: strength={self.strength:.2f}, "
                f"confidence={self.confidence:.2f}, "
                f"affected={self.affected_particles}"
            )