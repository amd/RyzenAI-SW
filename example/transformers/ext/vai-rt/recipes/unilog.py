from .tools.cmake_recipe import *
from pathlib import Path


class unilog(CMakeRecipe):

    def __init__(self):
        super().__init__('unilog')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/unilog.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_extra_args(self):
        args = ['-DBUILD_PYTHON=ON', '-DBUILD_TEST=ON']
        if self.is_windows():
            args.extend(['-DINSTALL_USER=ON'])
        return args
