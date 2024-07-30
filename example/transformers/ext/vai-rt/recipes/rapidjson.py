from .tools.cmake_recipe import *
import urllib.request
import zipfile


class rapidjson(CMakeRecipe):

    def __init__(self):
        super().__init__('rapidjson')

    def git_url(self):
        return "https://github.com/Tencent/rapidjson.git"

    def git_branch(self):
        return "master"

    def git_commit(self):
        return super().git_commit() or "06d58b9e848c650114556a23294d0b6440078c61"
