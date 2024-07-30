from pathlib import Path
import os
import logging
import multiprocessing
from . import workspace
from . import shell
from . import recipe
import platform
import shutil

home = Path.home()


class CMakeRecipe(recipe.Recipe):

    def __init__(self, name):
        super().__init__(name)

    def CMakeLists_txt_dir(self):
        return "."

    def cmake_extra_args(self):
        return []

    def build_static(self):
        if self.is_windows():
            return True
        else:
            return False

    def cmake_pic_options(self):
        return ['-DCMAKE_POSITION_INDEPENDENT_CODE=ON']

    def config(self):
        with shell.Cwd(self.workspace()):
            self._run(['cmake'] + self.cmake_config_args(), dry_run=False)

    def build(self):
        with shell.Cwd(self.workspace()):
            self._run(['cmake'] + self.cmake_build_args(), dry_run=False)

    def install(self):
        with shell.Cwd(self.workspace()):
            self._run(['cmake'] + self.cmake_install_args(), dry_run=False)

    def cmake_config_default_args(self):
        return ['-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DProtobuf_DEBUG=ON']

    def cmake_config_build_type(self):
        if self.is_windows():
            return '-DCMAKE_CONFIGURATION_TYPES=' + self.build_type(
            ).capitalize()
        else:
            return '-DCMAKE_BUILD_TYPE=' + self.build_type().capitalize()

    def cmake_use_ninja(self):
        return False

    def cmake_generateor(self):
        if self.cmake_use_ninja():
            return ['-G', 'Ninja']
        elif self.is_windows():
            return [
                '-A', 'x64', '-T', 'host=x64', '-G', 'Visual Studio 16 2019'
            ]
        else:
            return []

    def cmake_config_project_args(self):
        args = []
        if self.build_static():
            args.extend(['-DBUILD_SHARED_LIBS=OFF'])
        else:
            args.extend(['-DBUILD_SHARED_LIBS=ON'])

        args.extend(self.cmake_pic_options())
        args.extend([self.cmake_config_build_type()])
        args.extend(self.cmake_generateor())
        args.extend(['-DCMAKE_INSTALL_PREFIX=' + str(self.install_prefix())])
        args.extend(['-DCMAKE_PREFIX_PATH=' + str(self.install_prefix())])
        if self.is_crosss_compilation():
            args.extend([
                f"-DCMAKE_TOOLCHAIN_FILE={os.environ['OECORE_NATIVE_SYSROOT']}/usr/share/cmake/OEToolchainConfig.cmake"
            ])
            if self.build_type() == 'Release':
                args.extend(['-DCMAKE_CXX_FLAGS_RELEASE=-s'])

        args.extend(self.cmake_extra_args())
        args.extend(['-B', str(self.build_dir())])
        args.extend(['-S', str(self.src_dir() / self.CMakeLists_txt_dir())])
        return args

    def cmake_config_args(self):
        args = self.cmake_config_default_args()
        args.extend(self.cmake_config_project_args())
        return args

    def cmake_build_args(self):
        num_parallel_jobs = multiprocessing.cpu_count()
        args = [
            '--build',
            self.build_dir(),
            "-j",
            str(num_parallel_jobs),
            '--config',
            self.build_type(),
        ]
        if self.is_windows():
            args.extend([
                '--',
                # "/maxcpucount:{}".format(num_parallel_jobs),
                # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                # eventually, I found this flag works, we dont need also /maxcpucount
                "/p:CL_MPcount={}".format(num_parallel_jobs),
                "/nodeReuse:False",
            ])
        return args

    def cmake_install_args(self):
        args = ['--install', self.build_dir(), '--config', self.build_type()]
        return args

    def config(self):
        with shell.Cwd(self.workspace()):
            self._run(['cmake'] + self.cmake_config_args(), dry_run=False)
        if self.is_windows():
            # see https://github.com/mesonbuild/meson/issues/6882
            text_file = open(self.build_dir() / "Directory.Build.Props", "w")
            props = '''<Project>
  <PropertyGroup>
    <UseMultiToolTask>true</UseMultiToolTask>
    <EnforceProcessCountAcrossBuilds>true</EnforceProcessCountAcrossBuilds>
  </PropertyGroup>
</Project>
'''
            n = text_file.write(props)
            text_file.close()
            pass
        elif self.is_crosss_compilation():
            pass
        elif self.is_file(self.build_dir() / "compile_commands.json"):
            import shutil
            shutil.copyfile(self.build_dir() / "compile_commands.json",
                            self.src_dir() / "compile_commands.json")

    def build(self):
        if False and self.is_windows() and (not self.cmake_use_ninja()):
            # todo, msbuild is faster than 'cmake --build', I don't know why.
            num_parallel_jobs = multiprocessing.cpu_count()
            with shell.Cwd(self.build_dir()):
                self._run([
                    'msbuild',
                    f"{self.name()}.sln",
                    f"/p:Configuration={self.build_type()}",
                    "/maxcpucount:{}".format(num_parallel_jobs),
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                ])
        else:
            with shell.Cwd(self.workspace()):
                self._run(['cmake'] + self.cmake_build_args(), dry_run=False)

    def install(self):
        with shell.Cwd(self.workspace()):
            self._run(['cmake'] + self.cmake_install_args(), dry_run=False)

    def cmake(self):
        try:
            self._clean_log_file()
            self.config()
            self.build()
            self.install()
        finally:
            self._show_log_file()


def P(path):
    ret = Path(path).expanduser()
    if not ret.is_absolute():
        ret = ret.absolute()
    return ret


class _CMakeRecipeJson(CMakeRecipe):

    def __init__(self, workspace, project):
        super().__init__(project['name'])
        self._project = project
        self._workspace = workspace

    def workspace(self):
        ret = P(".")
        if 'VAI_RT_WORKSPACE' in os.environ:
            return Path(os.environ['VAI_RT_WORKSPACE'])
        elif 'workspace' in self._workspace:
            ret = P(self._workspace['workspace'])
        return ret

    def install_prefix(self):
        if "VAI_RT_PREFIX" in os.environ:
            return Path(os.environ.get("VAI_RT_PREFIX"))
        elif 'install_prefix' in self._workspace:
            return P(self._workspace['install_prefix'])
        else:
            return super().install_prefix()

    def build_dir(self):
        ret = super().build_dir()
        if 'VAI_RT_BUILD_DIR' in os.environ:
            return Path(os.environ['VAI_RT_BUILD_DIR']) / self.name()
        elif 'build_dir' in self._workspace:
            ret = P(self._workspace['build_dir']) / Path(super().name())
        return ret

    def src_dir(self):
        assert 'src_dir' in self._project
        return P(self._project['src_dir'])

    def download(self):
        logging.info(
            f"cancel downloading, use src_dir defined in json {self.src_dir()}"
        )

    def cmake_extra_args(self):
        return self._project.get('cmake_extra_args', [])


def load(workspace, project):
    return _CMakeRecipeJson(workspace, project)
