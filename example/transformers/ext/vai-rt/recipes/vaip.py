from .tools.shell import *
from .tools.cmake_recipe import *
import os


class vaip(CMakeRecipe):

    def __init__(self):
        super().__init__('vaip')

    def config(self):
        try:
            import build.__main__
        except ImportError as e:
            self._run(["python", "-m", "pip", "install", "build"])
        if os.environ.get("WITH_TVM_AIE_COMPILER", "OFF") == "ON" and \
                "VAIP_TVM_INSTALL_PATH" not in os.environ:
            os.environ["VAIP_TVM_INSTALL_PATH"] = str(self.install_prefix())
        super().config()

    def git_url(self):
        server = "gitenterprise.xilinx.com"
        repo = "VitisAI/vaip.git"
        if self._prefer_https_git_url:
            return "https://" + server + "/" + repo
        return "git@" + server + ":" + repo

    def git_branch(self):
        return "dev"

    def cmake_extra_args(self):
        args = [
            '-DBUILD_PYTHON=ON',
            '-DBUILD_TEST=ON',
            '-DProtobuf_DEBUG=ON',
            '-DProtobuf_USE_STATIC_LIBS=ON',
            '-DWITH_XCOMPILER=' + os.environ.get('WITH_XCOMPILER', 'ON'),
            '-DWITH_TVM_AIE_COMPILER=' + os.environ.get('WITH_TVM_AIE_COMPILER', 'OFF'),
        ]

        if 'GRAPH_ENGINE_COMMIT_ID' in os.environ.keys():
            GIT_COMMIT_ID = os.environ.get("GRAPH_ENGINE_COMMIT_ID")
            if GIT_COMMIT_ID:
                args.append('-DGRAPH_ENGINE_HASH=' + GIT_COMMIT_ID)
        if self.is_windows():
            args.extend(['-DINSTALL_USER=ON'])
        if self.is_crosss_compilation():
            args.append('-DFIND_FINGERPRINT=ON')
        return args
