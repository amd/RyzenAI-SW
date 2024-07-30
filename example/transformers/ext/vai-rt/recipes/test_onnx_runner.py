from .tools.shell import *
from .tools.cmake_recipe import *


class test_onnx_runner(CMakeRecipe):

    def __init__(self):
        super().__init__('test_onnx_runner')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/test_onnx_runner.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_use_ninja(self):
        return False
