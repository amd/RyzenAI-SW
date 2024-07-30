from .tools.cmake_recipe import *
from pathlib import Path


class zlib(CMakeRecipe):

    def __init__(self):
        super().__init__('zlib')

    def git_url(self):
        return "https://github.com/madler/zlib.git"

    def git_branch(self):
        return "v1.2.13"

    def git_commit(self):
        return super().git_commit() or "v1.2.13"

    def cmake_extra_args(self):
        return ['-DWITH_GFLAGS=OFF', '-DWITH_GTEST=OFF']

    def manifest_dir(self):
        return self.build_dir()
