from .tools.shell import *
from .tools.cmake_recipe import *


class tvm_engine(CMakeRecipe):

    def __init__(self):
        super().__init__('tvm-engine')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/tvm-engine.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "main"

    def cmake_extra_args(self):
        args = ['-DBUILD_TESTING=OFF']
        return args
