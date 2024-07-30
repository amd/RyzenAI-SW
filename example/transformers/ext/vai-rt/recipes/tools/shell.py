import logging
import shlex
import subprocess
import os
import sys
import traceback


def run(args, dry_run=False, quiet=False, output="build.log"):
    msg = f"running@[ {os.getcwd()} ] : " + " ".join(
        [shlex.quote(str(arg)) for arg in args])
    if not quiet:
        logging.info(msg)
    if not dry_run:
        with open(output, 'a') as file:
            file.write(msg)
            file.write("\n")
            file.flush()
            try:
                #below 3.8 see https://gitenterprise.xilinx.com/VitisAI/cmake.sh/issues/23
                if os.environ.get(
                        "BUILD_ID"
                ):  ## if running inside jenkins, do not rediret to log
                    subprocess.check_call([str(arg) for arg in args])
                else:
                    subprocess.check_call([str(arg) for arg in args],
                                          stdout=file,
                                          stderr=file)
            except subprocess.CalledProcessError as e:
                logging.error(f"command failure: {e.cmd}")
                raise


class Cwd(object):

    def __init__(self, cwd):
        self._old_cwd = os.getcwd()
        self._cwd = cwd

    def __enter__(self):
        os.chdir(self._cwd)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        os.chdir(self._old_cwd)
        if exc_type is not None:
            # traceback.print_exception(exc_type, exc_value, tb)
            return False
        return True
