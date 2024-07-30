from .tools.cmake_recipe import *


class vairt(CMakeRecipe):

    def __init__(self):
        super().__init__('vairt')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/vairt.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"
