from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Config:
    seed: int = 0
    map_size: int = 256
    train_maps: int = 1  # single-map learnability first; explicitly opt into up to ten later
    validation_maps: int = 5
    test_maps: int = 5
    map_family: str = "legacy"  # preserve the committed single-map experiment
    map_seed: int = 20261011  # dataset seed, independent of the learner seed
    map_order: str = "cyclic"
    sensor_implementation: str = "legacy"
    robots: int = 2
    sensor_range: int = 70
    grid_points: int = 14
    nodes: int = 208
    neighbors: int = 12
    input_dim: int = 12
    embedding_dim: int = 64
    dt: float = 1.0
    speed: float = 12.0  # map units / second; these synthetic maps have no metre calibration
    communication_range: float = 65.0
    bytes_per_second: float = 2048.0
    max_packet_bytes: int = 256
    tile_size: int = 16
    pose_period: float = 4.0
    estimated_contact_seconds: float = 4.0
    max_wait_seconds: float = 8.0
    horizon_seconds: float = 256.0
    completion: float = 0.99
    variant: str = "B0"  # B0, A, B, AB
    content: str = "fifo"  # fifo, rule, learned
    coupled: bool = True
    gamma: float = 0.995
    learning_rate: float = 1e-4
    policy_learning_rate: Optional[float] = None  # None uses the common Actor/Q rate
    alpha_learning_rate: float = 1e-4
    batch_size: int = 16
    replay_size: int = 3000
    compact_replay: bool = False  # lossless bit packing of binary adjacency matrices
    warmup: int = 128
    updates_per_decision: int = 1
    delivery_reward: float = 1.0
    coverage_reward: str = "union"  # explicitly test local_mean for the per-robot completion task
    pointer_mode: str = "centered"  # fresh runs use the stability fix
    time_penalty: float = 0.02
    label_sampling: str = "spread"  # cover the whole queue, not just its old prefix
    content_score_bound: str = "sender"  # legal novelty is an observable upper bound
    delivery_queue_order: str = "service"  # a partially sent map must finish first
    horizon_target: str = "terminal"  # the observed task deadline ends future returns
    frontier_shaping_scale: float = 0.0  # optional local potential, common to all variants
    success_imitation_weight: float = 0.0  # self-generated successful training episodes only
    success_episode_limit: int = 3
    success_memory_mode: str = "fastest"  # per_map retains one successful trace per layout
    success_sampling: str = "pooled"  # optional equal-map, physical-time tail emphasis
    torch_threads: int = 2

    def validate(self):
        if self.robots != 2:
            raise ValueError("This feasibility profile supports exactly two robots.")
        if self.variant not in {"B0", "A", "B", "AB"}:
            raise ValueError("Unknown ablation variant")
        if self.content not in {"fifo", "rule", "learned"}:
            raise ValueError("Unknown packet selection mode")
        if self.coverage_reward not in {"union", "local_mean"}:
            raise ValueError("Unknown coverage reward")
        if self.pointer_mode not in {"legacy", "centered"}:
            raise ValueError("Unknown pointer mode")
        if self.time_penalty < 0:
            raise ValueError("Time penalty must be nonnegative")
        if min(self.learning_rate, self.alpha_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.policy_learning_rate is not None and self.policy_learning_rate <= 0:
            raise ValueError("Actor learning rate must be positive")
        if self.label_sampling not in {"prefix", "spread"}:
            raise ValueError("Unknown content label sampling mode")
        if self.content_score_bound not in {"none", "sender"}:
            raise ValueError("Unknown content score bound")
        if self.delivery_queue_order not in {"legacy", "service"}:
            raise ValueError("Unknown delivery queue order")
        if self.horizon_target not in {"terminal", "bootstrap"}:
            raise ValueError("Unknown task deadline target")
        if self.frontier_shaping_scale < 0:
            raise ValueError("Frontier shaping scale must be nonnegative")
        if self.frontier_shaping_scale and self.horizon_target != "terminal":
            raise ValueError("Finite-task potential shaping requires terminal deadline targets")
        if self.success_imitation_weight < 0 or self.success_episode_limit < 1:
            raise ValueError("Success imitation needs a nonnegative weight and positive episode limit")
        if self.success_memory_mode not in {"fastest", "per_map"}:
            raise ValueError("Unknown success memory mode")
        if self.success_sampling not in {"pooled", "map_tail"}:
            raise ValueError("Unknown success demonstration sampling")
        if self.success_sampling == "map_tail" and self.success_memory_mode != "per_map":
            raise ValueError("Equal-map tail sampling requires per-map success memory")
        if self.success_memory_mode == "per_map" and self.success_episode_limit < self.train_maps:
            raise ValueError("Per-map success memory needs at least one slot per training map")
        if min(self.dt, self.speed, self.bytes_per_second, self.horizon_seconds,
               self.communication_range, self.estimated_contact_seconds) <= 0:
            raise ValueError("Time, speed, capacity and range must be positive")
        if min(self.train_maps, self.validation_maps, self.test_maps) < 1:
            raise ValueError("Every map split must be nonempty")
        if self.map_family not in {"legacy", "simple_rooms_v1"}:
            raise ValueError("Unknown map family")
        if self.map_order not in {"cyclic", "shuffled"}:
            raise ValueError("Unknown training map order")
        if self.sensor_implementation not in {"legacy", "vectorized"}:
            raise ValueError("Unknown sensor implementation")
        limits = (10, 5, 5) if self.map_family == "legacy" else (10000, 10000, 10000)
        if any(n > limit for n, limit in zip(
                (self.train_maps, self.validation_maps, self.test_maps), limits)):
            raise ValueError("Map counts exceed this family's split limits")
        if not 0 <= self.map_seed < 2 ** 32 or not 0 <= self.seed < 2 ** 32:
            raise ValueError("Seeds must be unsigned 32-bit integers")
        if self.grid_points ** 2 + 1 > self.nodes:
            raise ValueError("Node padding must fit the complete simple graph")
        if self.map_size != 256:
            raise ValueError("The initial simple-map generator uses 256x256 maps")
        if self.neighbors < 2 or not 0 < self.completion <= 1:
            raise ValueError("Invalid graph or completion setting")
        if self.embedding_dim % 8:
            raise ValueError("IR2 attention requires embedding_dim divisible by eight")
        if self.max_packet_bytes < 32 or self.max_packet_bytes > 512:
            raise ValueError("Packet size must be between 32 and 512 bytes")
        if min(self.batch_size, self.warmup, self.replay_size, self.updates_per_decision,
               self.pose_period, self.max_wait_seconds, self.tile_size, self.torch_threads) < 1:
            raise ValueError("Training sizes and time periods must be positive")
        if self.replay_size < max(self.batch_size, self.warmup):
            raise ValueError("Replay is too small to reach the warmup or batch size")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, values):
        # Missing fields mean the historical behavior, not today's fresh defaults.
        return cls(**{"pointer_mode": "legacy", "time_penalty": 0.02,
                     "label_sampling": "prefix", "content_score_bound": "none",
                     "delivery_queue_order": "legacy", "horizon_target": "bootstrap",
                     "frontier_shaping_scale": 0.0,
                     "success_imitation_weight": 0.0, "success_episode_limit": 3,
                     "coverage_reward": "union",
                     "policy_learning_rate": values.get("learning_rate", 1e-4),
                     "alpha_learning_rate": values.get("learning_rate", 1e-4), **values})
