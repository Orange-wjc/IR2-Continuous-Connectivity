"""Training entry: --finite-contact selects the reviewed native-map experiment."""
from pathlib import Path
import runpy
import sys

# CPU pool limit; pilot training uses one fresh rollout per method/update round.
# Label collection and evaluation use the full pool.
NUM_META_AGENT = 24


def main():
    if '--finite-contact' in sys.argv:
        sys.argv.remove('--finite-contact')
        from native_pilot import main as native_main
        native_main(default_num_meta_agent=NUM_META_AGENT)
    else:
        runpy.run_path(str(Path(__file__).with_name('driver_legacy.py')), run_name='__main__')


if __name__ == '__main__':
    main()
