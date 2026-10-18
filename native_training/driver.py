"""Frozen native experiment entry, with no upstream Ray startup."""
from native_driver import main
NUM_META_AGENT = 24
if __name__ == '__main__':
    main(default_num_meta_agent=NUM_META_AGENT)
