from .tools.shell import *
from .tools.cmake_recipe import *


class graph_engine(CMakeRecipe):

    def __init__(self):
        super().__init__('graph-engine')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/graph-engine.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_extra_args(self):
        return ['-DBUILD_TEST=OFF','-DBUILD_TESTS=OFF', '-DBUILD_DOC=OFF']

    def download(self):
        super().download()
        if not self.is_file(
                self.src_dir() / "external" / "concurrentqueue" / "README.md"):
            with shell.Cwd(self.src_dir()):
                self._run(["git", "submodule", "update", "--init"])
