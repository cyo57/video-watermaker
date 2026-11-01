"""Linux subprocess launcher: do not leave encoders behind after SIGKILL."""

import ctypes
import os
import signal
import sys

parent = int(sys.argv[1])
if sys.platform.startswith("linux"):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent:
        raise SystemExit(130)
os.execvp(sys.argv[2], sys.argv[2:])
