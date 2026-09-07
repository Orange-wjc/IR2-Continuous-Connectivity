"""Reproducible, disjoint map splits for the 400 hybrid maps."""

import random


HYBRID_SPLIT_SEED = 20260907
HYBRID_DEMO_MAP_ID = 1

_remaining_ids = list(range(2, 401))
random.Random(HYBRID_SPLIT_SEED).shuffle(_remaining_ids)

HYBRID_TRAIN_MAP_IDS = tuple(sorted(_remaining_ids[:300]))
HYBRID_VALIDATION_MAP_IDS = tuple(sorted(_remaining_ids[300:350]))
HYBRID_TEST_MAP_IDS = (HYBRID_DEMO_MAP_ID,) + tuple(sorted(_remaining_ids[350:]))

HYBRID_TRAIN_MAP_FILES = tuple(f"{map_id}.png" for map_id in HYBRID_TRAIN_MAP_IDS)
HYBRID_VALIDATION_MAP_FILES = tuple(f"{map_id}.png" for map_id in HYBRID_VALIDATION_MAP_IDS)
HYBRID_TEST_MAP_FILES = tuple(f"{map_id}.png" for map_id in HYBRID_TEST_MAP_IDS)

assert len(HYBRID_TRAIN_MAP_FILES) == 300
assert len(HYBRID_VALIDATION_MAP_FILES) == 50
assert len(HYBRID_TEST_MAP_FILES) == 50
assert not (set(HYBRID_TRAIN_MAP_FILES) & set(HYBRID_VALIDATION_MAP_FILES))
assert not (set(HYBRID_TRAIN_MAP_FILES) & set(HYBRID_TEST_MAP_FILES))
assert not (set(HYBRID_VALIDATION_MAP_FILES) & set(HYBRID_TEST_MAP_FILES))
assert "1.png" in HYBRID_TEST_MAP_FILES
