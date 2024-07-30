from .tools.cmake_recipe import *


class vart(CMakeRecipe):

    def __init__(self):
        super().__init__('vart')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/vart.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_extra_args(self):
        args = ['-DBUILD_PYTHON=ON', '-DBUILD_TEST=ON', '-DENABLE_CPU_RUNNER=ON', '-DENABLE_DPU_RUNNER=ON']
        if self.is_windows():
            args.remove('-DBUILD_TEST=ON')
            args.extend(['-DINSTALL_USER=ON', '-DBUILD_TEST=OFF'])
        return args
