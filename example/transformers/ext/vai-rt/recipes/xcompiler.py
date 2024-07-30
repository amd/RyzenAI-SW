from .tools.shell import *
from .tools.cmake_recipe import *
from pathlib import Path


class xcompiler(CMakeRecipe):

    def __init__(self):
        super().__init__('xcompiler')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/xcompiler.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_extra_args(self):
        args = [
            '-DBUILD_PYTHON=OFF', '-DBUILD_TEST=OFF',
            '-DINSTALL_FAILURE_FUNC=OFF',
            '-DProtobuf_USE_STATIC_LIBS=ON'
        ]
        if not self.is_crosss_compilation():
            args.append('-DENABLE_IPU=ON')
        if self.build_type() == "Debug":
            args.append('-DSKIP_CLEAN_GRAPH=ON')
        return args
