"""Deterministic, disjoint simple layouts, without reading the large DungeonMaps set."""
import hashlib
import numpy as np
from scipy.ndimage import label


def simple_map(split, index):
    if split not in {"train", "validation", "test"} or index < 0:
        raise ValueError("Invalid map key")
    # Disjoint door positions encode 10 train + 5 validation + 5 test geometries.
    offsets = {"train": 0, "validation": 10, "test": 15}
    limits = {"train": 10, "validation": 5, "test": 5}
    if index >= limits[split]:
        raise ValueError("Map index exceeds the fixed feasibility split")
    key = offsets[split] + index
    truth = np.ones((256, 256), dtype=np.uint8)
    truth[16:240, 16:240] = 255
    wall_x = 119 + key % 5
    door_y = 40 + key * 7
    truth[16:240, wall_x:wall_x + 8] = 1
    truth[door_y:door_y + 25, wall_x:wall_x + 8] = 255
    # A short alcove varies geometry but leaves both rooms connected.
    truth[190:198, 150:178 + key % 7] = 1
    starts = np.array([[wall_x - 28, door_y + 8],
                       [wall_x - 28, door_y + 20]], dtype=np.float64)
    components, count = label(truth == 255)
    if count != 1 or any(truth[int(y), int(x)] != 255 for x, y in starts):
        raise RuntimeError("Simple map is not reachable from its starting positions")
    return truth, starts, {
        "split": split, "index": index, "layout_id": key,
        "sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "units": "map_units", "starts": starts.tolist(),
    }


def diverse_map(split, index, dataset_seed=20261011):
    """IID split-specific layouts; local RNG never advances training RNG streams."""
    if split not in {"train", "validation", "test"} or not 0 <= index < 10000:
        raise ValueError("Invalid diverse-map key")
    key = f"simple_rooms_v1:{dataset_seed}:{split}:{index}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    left, top = rng.randint(12, 25, size=2)
    right, bottom = rng.randint(232, 245, size=2)
    rooms = 3 if rng.rand() < .25 else 2
    walls = ([int(rng.randint(92, 161))] if rooms == 2 else
             [int(rng.randint(78, 101)), int(rng.randint(155, 179))])
    thickness = int(rng.randint(4, 9))
    truth = np.ones((256, 256), dtype=np.uint8)
    truth[top:bottom, left:right] = 255
    doors = []
    for wall in walls:
        height = int(rng.randint(32, 45))
        y = int(rng.randint(top + 20, bottom - height - 19))
        truth[top:bottom, wall:wall + thickness] = 1
        truth[y:y + height, wall:wall + thickness] = 255
        doors.append([wall, y, thickness, height])
    center = doors[0][1] + doors[0][3] // 2
    x = walls[0] - int(rng.randint(24, 37))
    starts = np.array([[x, center - 6], [x, center + 6]], dtype=np.float64)
    obstacles = []
    boundaries = [left] + [w + thickness for w in walls]
    ends = walls + [right]
    for room in range(rooms):
        if rng.rand() < .65:
            width, height = int(rng.randint(10, 23)), int(rng.randint(8, 21))
            x0 = int(rng.randint(boundaries[room] + 12, ends[room] - width - 11))
            y0 = int(rng.randint(top + 16, bottom - height - 15))
            # Keep a generous spawn corridor; no hidden repair changes geometry.
            near_start = any(x0 - 14 <= sx <= x0 + width + 14 and
                             y0 - 14 <= sy <= y0 + height + 14 for sx, sy in starts)
            if not near_start:
                truth[y0:y0 + height, x0:x0 + width] = 1
                obstacles.append([x0, y0, width, height])
    rotation = int(rng.randint(4))
    truth = np.rot90(truth, rotation).copy()
    for _ in range(rotation):
        starts = np.column_stack((starts[:, 1], 255 - starts[:, 0]))
    _, components = label(truth == 255)
    if components != 1 or any(truth[int(y), int(x)] != 255 for x, y in starts):
        raise RuntimeError(f"Invalid generated map: {key}")
    canonical = min(hashlib.sha256(np.rot90(a, k).tobytes()).hexdigest()
                    for a in (truth, np.fliplr(truth)) for k in range(4))
    return truth, starts, {
        "split": split, "index": index, "layout_id": key, "family": "simple_rooms_v1",
        "dataset_seed": dataset_seed, "sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "canonical_sha256": canonical, "units": "map_units", "starts": starts.tolist(),
        "rooms": rooms, "rotation": rotation, "bounds_before_rotation":
        [int(left), int(top), int(right), int(bottom)], "doors_before_rotation": doors,
        "obstacles_before_rotation": obstacles,
    }


def configured_map(config, split, index):
    limit = {"train": config.train_maps, "validation": config.validation_maps,
             "test": config.test_maps}.get(split, 0)
    if not 0 <= index < limit:
        raise ValueError("Map index exceeds the configured split")
    if config.map_family == "legacy":
        return simple_map(split, index)
    if config.map_family == "simple_rooms_v1":
        return diverse_map(split, index, config.map_seed)
    raise ValueError("Unknown map family")


def training_map_index(config, episode):
    """One deterministic permutation per sweep, independent of policy sampling RNG."""
    if episode < 1:
        raise ValueError("Episodes are one-based")
    cycle, position = divmod(episode - 1, config.train_maps)
    if config.map_order == "cyclic":
        return position
    seed = (config.seed + 1000003 * cycle) % (2 ** 32)
    return int(np.random.RandomState(seed).permutation(config.train_maps)[position])


def manifest(config):
    return [configured_map(config, split, i)[2]
            for split, count in (("train", config.train_maps),
                                 ("validation", config.validation_maps),
                                 ("test", config.test_maps))
            for i in range(count)]
