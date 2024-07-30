from .tools.shell import *
from .tools.cmake_recipe import *


class opencv(CMakeRecipe):

    def __init__(self):
        super().__init__('opencv')

    def git_url(self):
        return "https://github.com/opencv/opencv.git"

    def git_branch(self):
        return "4.6.0"

    def git_commit(self):
        return super().git_commit() or "4.6.0"

    def cmake_extra_args(self):
        "opencv use CMAKE_BUILD_TYPE on windows also"
        "opencv in Python is not used in the project and requires Admin right to install"
        "And setting PYTHONPATH can not overwrite FindPythonLib's returning path used in OpenCV repo"
        args = ["-DCMAKE_BUILD_TYPE=" + self.build_type(), '-DBUILD_opencv_python2=OFF', '-DBUILD_opencv_python3=OFF']
        if self.build_static():
            # use /MDd instread of /MTd
            args.append('-DBUILD_WITH_STATIC_CRT=OFF')
        return args
    
    def manifest_dir(self):
        return self.build_dir()
