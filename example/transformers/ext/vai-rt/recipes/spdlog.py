from .tools.cmake_recipe import *
from pathlib import Path


class spdlog(CMakeRecipe):

    def __init__(self):
        super().__init__('spdlog')

    def git_url(self):
        return "https://github.com/gabime/spdlog.git"

    def git_branch(self):
        return "v1.x"

    def git_commit(self):
        return super().git_commit() or "v1.x"

    def cmake_extra_args(self):
        return ['-DWITH_GFLAGS=OFF', '-DWITH_GTEST=OFF']

    def manifest_dir(self):
        return self.build_dir()
