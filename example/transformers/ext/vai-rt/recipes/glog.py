from .tools.cmake_recipe import *
from pathlib import Path


class glog(CMakeRecipe):

    def __init__(self):
        super().__init__('glog')

    def git_url(self):
        return "https://github.com/google/glog.git"

    def git_branch(self):
        return "v0.6.0"

    def git_commit(self):
        return super().git_commit() or "v0.6.0"

    def git_patch(self):
        return [
            Path(__file__).parent / ".." / "patch" / "glog_enable_fatal_throw_exception.patch"
        ]

    def cmake_extra_args(self):
        return ['-DWITH_GFLAGS=OFF', '-DWITH_GTEST=OFF']

    def manifest_dir(self):
        return self.build_dir()
