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


def manifest(config):
    return [simple_map(split, i)[2]
            for split, count in (("train", config.train_maps),
                                 ("validation", config.validation_maps),
                                 ("test", config.test_maps))
            for i in range(count)]
