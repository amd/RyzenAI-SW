from .tools.cmake_recipe import *


class gtest(CMakeRecipe):

    def __init__(self):
        super().__init__('gtest')

    def git_url(self):
        return "https://github.com/google/googletest.git"

    def git_branch(self):
        return "release-1.12.1"
    
    def git_commit(self):
        return super().git_commit() or "release-1.12.1"
    
    def cmake_extra_args(self):
        return ['-Dgtest_force_shared_crt=ON']