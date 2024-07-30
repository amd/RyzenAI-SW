from .tools.cmake_recipe import *
import multiprocessing
import urllib.request
import os
import time
import shutil
import zipfile
import subprocess
import sys


class xrt(CMakeRecipe):

    def __init__(self):
        super().__init__('xrt')

    def git_url(self):
        return "https://github.com/Xilinx/XRT.git"

    def git_branch(self):
        return "2022.2"

    def git_commit(self):
        return super().git_commit(
        ) or "abcdeecb6197b51bf29274a91323c26801deddcc"

    def need_install(self):
        return not os.path.isdir(
            self.install_prefix() / "opt" / "xilinx" / "xrt")

    def proj_build_dir(self):
        return self.src_dir() / "build"

    def release_dir(self):
        return self.proj_build_dir() / self.build_type() / "opt"

    def install_path(self):
        return self.install_prefix() / "opt"

    def make_all(self):
        if not self.need_install():
            logging.info(
                f"\t{self.name()} is already installed under {self.install_prefix()}"
            )
            return
        try:
            self.report()
            start = time.time()
            self._clean_log_file()
            self.download()
            if self.build_type() == "Debug":
                build_type = "-dbg"
            else:
                build_type = "-opt"
            with shell.Cwd(self.proj_build_dir()):
                self._run([
                    "env", "CMAKE_PREFIX_PATH=" + str(self.install_prefix()),
                    "CPLUS_INCLUDE_PATH=" + str(self.install_prefix()) +
                    "/include",
                    "LD_LIBRARY_PATH=" + str(self.install_prefix()) + "/lib",
                    "./build.sh", build_type, "-noctest", "-j",
                    str(multiprocessing.cpu_count())
                ])
            if os.path.exists(self.install_path()):
                shutil.rmtree(self.install_path())
            shutil.copytree(self.release_dir(), self.install_path())
        except subprocess.CalledProcessError as e:
            logging.error(
                f"!!! failure :( !!!  build [{self.name()}] failed!. cmd= {e.cmd} log {self.log_file()} are show as below:"
            )
            self._show_log_file()
            sys.exit(1)
        else:
            end = time.time()
            elapse = "{:.2f}".format(end - start)
            logging.info(
                f"=== end :) ==== build [{self.name()}] {elapse} seconds done.\n\tplease read {self.log_file()} for details"
            )
        finally:
            pass
