import sys
from multiprocessing import freeze_support

from geofem_app.cli import main


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main(sys.argv[1:] or ["gui"]))
