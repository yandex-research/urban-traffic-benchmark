__doc__ = """Internal infrastructure dependencies"""

from typing import List, Any, Dict
from distutils.dir_util import copy_tree
import os

try:
    import nirvana_dl
except ImportError:
    nirvana_dl = None



def copy_snapshot_to_out(out):
    """
    The preempted run transfers its "state" to the restarted run through "snapshot path".
    "state" is a tar-archive that contains all files put into "snapshot path" by the preempted run.
    This function moves the files in the "state" archive to you local "out" dir.
    """
    out = str(out)

    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy the previous state from {snapshot_path} to {out}")
        copy_tree(snapshot_path, out)
        # os.system(f"tar -xf {out}/state -C {out}/")
    if (x := os.environ.get("SNAPSHOT_PATH")):
        print(f"Copy the previous state from {x} to {out}")
        copy_tree(x, out)


def copy_out_to_snapshot(out, dump=True):
    """This function copies all files in the local "out" directory to "snapshot path".
    dump: If True, put these files into tar-archive "state" and
          send it to the Python DL output.
    """
    out = str(out)

    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy {out} to the snapshot path: {snapshot_path}")

        os.system(f"rm {snapshot_path}/state")
        copy_tree(out, snapshot_path, update=1)

        if dump:
            nirvana_dl.snapshot.dump_snapshot(snapshot_path)
