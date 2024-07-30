import os
import platform
from pathlib import Path
import shlex
import subprocess
from . import shell
import logging

home = Path.home()


class Workspace(object):

    def __init__(self):
        super().__init__()
        self._dry_run = False

    def _run(self, args, dry_run=False):
        return shell.run(args,
                         output=self.log_file(),
                         quiet=False,
                         dry_run=self._dry_run)

    def run(self, args, dry_run=False):
        msg = f"running@[ {os.getcwd()} ] : " + " ".join(
            [shlex.quote(str(arg)) for arg in args])
        logging.info(msg)
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as e:
            logging.error(f"command failure: {e.cmd}")
            raise

    def user(self):
        if self.is_windows():
            return os.environ['USERNAME']
        else:
            return os.environ['USER']

    def is_crosss_compilation(self):
        return 'OECORE_TARGET_SYSROOT' in os.environ

    def is_windows(self):
        return platform.system() == "Windows"

    def home(self):
        return Path.home()

    def workspace(self):
        self.home() / "workspace"

    def build_dir(self):
        return self.home() / "build"

    def log_file(self):
        os.makedirs(self.build_dir(), exist_ok=True)
        return self.build_dir() / (self.name() + ".log")

    def _clean_log_file(self):
        if os.path.exists(self.log_file()):
            os.remove(self.log_file())
        os.makedirs(self.workspace(), exist_ok=True)
        os.makedirs(self.build_dir(), exist_ok=True)
        with open(self.log_file(), "w") as file:
            file.write("start to build " + self.name())

    def _show_log_file(self):
        with open(self.log_file(), "r") as file:
            all_of_it = file.read()
            print(all_of_it)

    def is_dir(self, dir_name):
        return os.path.isdir(dir_name)

    def is_file(self, file_name):
        return os.path.isfile(file_name)

    @property
    def dry_run(self):
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value):
        self._dry_run = value
