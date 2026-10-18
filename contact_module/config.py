"""Explicit experiment configuration; no environment/model imports."""
from dataclasses import asdict, dataclass
import math
import os


@dataclass(frozen=True)
class ContactConfig:
    method: str = 'original'
    bytes_per_second: float = 2048.
    packet_bytes: int = 512
    tile_size: int = 16
    pose_period: float = 4.
    contact_seconds: float = 4.
    speed: float = 12.
    wait_seconds: float = 1.
    time_penalty: float = .02
    coupled: bool = True

    @property
    def enabled(self):
        return self.method != 'original'

    @property
    def learned(self):
        return self.method == 'finite_learned'

    def to_dict(self):
        return asdict(self)

    def validate_checkpoint(self, state):
        previous = state.get('contact_config', {'method':'original'})
        if previous.get('method') != self.method:
            raise ValueError('Checkpoint method differs; use a separate IR2_RUN_NAME and fresh training')
        if self.enabled and previous != self.to_dict():
            raise ValueError('Cannot silently change finite-contact configuration on resume')
        if self.learned and 'contact_learning' not in state:
            raise ValueError('Learned checkpoint is missing the auxiliary head/optimizer state')

    @classmethod
    def from_env(cls):
        value = cls(method=os.environ.get('IR2_METHOD', 'original'),
                    bytes_per_second=float(os.environ.get('IR2_BYTES_PER_SECOND', '2048')),
                    coupled=os.environ.get('IR2_CONTACT_COUPLED', '1') == '1')
        if value.method not in ('original', 'finite_fifo', 'finite_rule', 'finite_learned'):
            raise ValueError('Unknown IR2_METHOD: ' + value.method)
        if not math.isfinite(value.bytes_per_second) or value.bytes_per_second <= 0:
            raise ValueError('IR2_BYTES_PER_SECOND must be finite and positive')
        return value
