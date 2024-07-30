from .tools.cmake_recipe import *
from pathlib import Path


class gsl(CMakeRecipe):
    def __init__(self):
        super().__init__("gsl")

    def git_url(self):
        return "https://github.com/microsoft/GSL.git"

    def git_branch(self):
        return "v4.0.0"

    def git_commit(self):
        return super().git_commit() or "v4.0.0"

    def git_patch(self):
        return [Path(__file__).parent / ".." / "patch" / "gsl_v4.0.0_1064.patch"]

    def manifest_dir(self):
        return self.build_dir()
