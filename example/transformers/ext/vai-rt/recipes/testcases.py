from .tools.cmake_recipe import *


class testcases(CMakeRecipe):

    def __init__(self):
        super().__init__('testcases')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/testcases.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"
