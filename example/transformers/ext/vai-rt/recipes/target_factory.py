from .tools.cmake_recipe import *


class target_factory(CMakeRecipe):

    def __init__(self):
        super().__init__('target_factory')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/target_factory.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_generateor(self):
        return []

    def cmake_extra_args(self):
        args = [
            '-DBUILD_TEST=ON',
            "-DProtobuf_USE_STATIC_LIBS=ON"
        ]
        if not self.is_crosss_compilation():
            args.append('-DENABLE_IPU=ON')

        return args
