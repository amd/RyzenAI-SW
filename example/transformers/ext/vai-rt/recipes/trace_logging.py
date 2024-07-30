from .tools.cmake_recipe import *


class trace_logging(CMakeRecipe):

    def __init__(self):
        super().__init__('trace_logging')

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/trace-logging.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"
