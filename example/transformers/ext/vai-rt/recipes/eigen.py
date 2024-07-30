from .tools.cmake_recipe import *


class eigen(CMakeRecipe):

    def __init__(self):
        super().__init__('eigen')

    def git_url(self):
        return "https://gitlab.com/libeigen/eigen.git"

    def git_commit(self):
        return super().git_commit() or "d10b27fe37736d2944630ecd7557cefa95cf87c9"

    def manifest_dir(self):
        return self.build_dir()
