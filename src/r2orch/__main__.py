import os
import sys
import traceback

from .cli import main

try:
    code = main()
except SystemExit as exc:
    code = exc.code
except BaseException:
    traceback.print_exc()
    code = 1
# Background collection/evaluation threads are not daemons; a crash in the main thread must
# not leave the process (and the driver waiting on it) alive.
sys.stdout.flush()
sys.stderr.flush()
os._exit(code if isinstance(code, int) else (0 if code is None else 1))
