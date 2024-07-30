from .tools.cmake_recipe import *


class pybind11(CMakeRecipe):

    def __init__(self):
        super().__init__('pybind11')

    def git_url(self):
        return "https://github.com/pybind/pybind11.git"

    def git_branch(self):
        return "v2.10.0"

    def git_commit(self):
        return super().git_commit() or "v2.10.0"

    def cmake_extra_args(self):
        return ['-DBUILD_TESTING=OFF']
