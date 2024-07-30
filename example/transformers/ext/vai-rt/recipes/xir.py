from .tools.shell import *
from .tools.cmake_recipe import *


class xir(CMakeRecipe):

    def __init__(self):
        super().__init__('xir')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/xir.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_extra_args(self):
        args = [
            '-DBUILD_PYTHON=ON', '-DProtobuf_DEBUG=ON',
            '-DProtobuf_USE_STATIC_LIBS=ON', '-DIS_DISTRIBUTION=ON'
        ]
        if self.is_windows():
            args.extend(['-DINSTALL_USER=ON'])
        return args
