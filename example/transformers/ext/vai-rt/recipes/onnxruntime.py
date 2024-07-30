from .tools.shell import *
from .tools.recipe import *
import multiprocessing
from pathlib import Path
import logging
import os


class onnxruntime(Recipe):

    def __init__(self):
        super().__init__('onnxruntime')

    def git_url(self):
        return "https://github.com/microsoft/onnxruntime.git"

    def git_branch(self):
        return "v1.15.1"

    def config(self):
        pass

    def workspace(self):
        if self.is_windows():
            if "VAI_RT_WORKSPACE" not in os.environ:
                # on windows 'build.py' does not support network drive mapping.
                return Path.home() / "workspace"
            else:
                return super().workspace()
        else:
            return super().workspace()

    def install_pip_and_packaging(self):
        chmod_cmd = [
            "sudo", "chmod", "777", "-R",
            f"{os.environ['OECORE_NATIVE_SYSROOT']}"
        ]
        wget_cmd = [
            "wget", "-P", "/tmp", "https://bootstrap.pypa.io/get-pip.py"
        ]
        install_cmd = ["python", "/tmp/get-pip.py"]
        packaging_cmd = ["python", "-m", "pip", "install", "packaging"]
        self._run(chmod_cmd)
        self._run(wget_cmd)
        self._run(install_cmd)
        self._run(packaging_cmd)

    def build(self):
        with shell.Cwd(self.src_dir()):
            os.makedirs(self.build_dir(), exist_ok=True)
            if not self.is_file(self.src_dir() / "cmake" / "external" /
                                "onnx" / "CMakeLists.txt"):
                self._run(["git", "submodule", "update", "--init"])
            cmd = [
                "python",
                "tools/ci_build/build.py",
                *[x for x in ["--cmake_generator", "Visual Studio 16 2019"] if self.is_windows()],
                "--use_vitisai",
                "--disable_memleak_checker",
                "--build_shared_lib",
                "--skip_submodule_sync",
                "--enable_pybind",
                "--build_wheel",
                "--config",
                self.build_type(),
                "--parallel",
                str(multiprocessing.cpu_count()),
                "--build_dir",
                str(self.build_dir()),
                "--skip_tests",
                "--cmake_extra_defines",
                "CMAKE_INSTALL_PREFIX=" + str(self.install_prefix()),
                'CMAKE_EXPORT_COMPILE_COMMANDS=ON',
                'CMAKE_TLS_VERIFY=FALSE',
                'BUILD_ONNX_PYTHON=ON',
                "CMAKE_PREFIX_PATH=" + str(self.install_prefix()),
            ]
            # used for CI only, Intel CPU's SSE AVX is not using IEEE standard and may results in different value
            if (os.environ.get("onnxruntime_DONT_VECTORIZE", "OFF") == "ON"):
                cmd.extend(["onnxruntime_DONT_VECTORIZE=1"])
            if self.is_crosss_compilation():
                if self.build_type() == 'Release':
                    cmd.extend(['CMAKE_CXX_FLAGS_RELEASE=-s'])
                sdk_py_version = '3.9'
                if os.environ['OECORE_SDK_VERSION'] == '2023.1':
                    sdk_py_version = '3.10'
                cmd.extend([
                    f"CMAKE_TOOLCHAIN_FILE={os.environ['OECORE_NATIVE_SYSROOT']}/usr/share/cmake/OEToolchainConfig.cmake",
                    "onnxruntime_CROSS_COMPILING=ON",
                    "onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=ON",
                    f"Python_NumPy_INCLUDE_DIR={os.environ['OECORE_NATIVE_SYSROOT']}/usr/lib/python{sdk_py_version}/site-packages/numpy/core/include",
                    "protobuf_BUILD_PROTOC_BINARIES=OFF",
                    "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER",
                ])
                import shutil
                protoc = shutil.which("protoc")
                if not protoc is None:
                    cmd.extend(["--path_to_protoc_exe", protoc])
                # install pip on crosss_compilation
                try:
                    import pip
                except ImportError:
                    self.install_pip_and_packaging()
            else:
                try:
                    from wheel.bdist_wheel import bdist_wheel
                except ImportError:
                    update_whell_cmd = [
                        "python", "-m", "pip", "install", "wheel>=0.35.1"
                    ]
                    self._run(update_whell_cmd)
            self._run(cmd)
            if self.is_file(self.build_dir() / self.build_type() /
                            "compile_commands.json"):
                import shutil
                shutil.copyfile(
                    self.build_dir() / self.build_type() /
                    "compile_commands.json",
                    self.src_dir() / "compile_commands.json")

            self._run([
                "cmake",
                "--install",
                str(self.build_dir() / self.build_type()),
                "--config",
                self.build_type(),
            ])

    def _install_dist_whl(self, dst_path):
        if self.is_windows():
            dst_path = self.build_dir() / self.build_type() / self.build_type(
            ) / "dist"
        import site
        dst = Path(site.getusersitepackages())
        if self.is_crosss_compilation():
            dst = self.install_prefix() / "lib" / "python3.9" / "site-packages"

        whl_files = list(dst_path.glob('*.whl'))
        if whl_files:
            for whl_path in whl_files:
                self._run([
                    "python", "-m", "pip", "install", "--upgrade",
                    "--force-reinstall", whl_path, "--target", dst
                ])
        else:
            logging.error(f"cannot find any whl file in {dst_path}")

    def install(self):
        if self.is_crosss_compilation():
            return
        if self.is_windows():
            build_dir = self.build_dir() / self.build_type() / self.build_type(
            )
        else:
            build_dir = self.build_dir() / self.build_type()

        self._run([
            "python", "-m", "pip", "install", "-r",
            build_dir / "requirements.txt"
        ])

        self._install_dist_whl(build_dir / "dist")

    def manifest_dir(self):
        return self.build_dir() / self.build_type()

    def git_patch(self):
        return [
            Path(__file__).parent / ".." / "patch" /
            "onnx_command_over_8192_char.patch",
            Path(__file__).parent / ".." / "patch" /
            "onnx_fp16_unsupported_on_a72_a53.patch",
            "272aab4afa92fedca5b537c78667d344a57b2325", # Fix-issues-on-Windows-for-Vitis-AI
            "6361b221033f04df2b38fded82030b89fd73ffa0", # vitis-ai-support-generic-data-type
            "b8bbc898c6436076f2852a461cd273ecf2bca805", # fix-errors-for-node-with-empty-name-for-vitis-ai
            "e951f837e43369a71663fe67beee8165397a93bd", # VITISAI-fix-out-of-bound-error-on-graph-with-loop
            "249c2221b69e3ac9d7bcd9e9d0455b59a4de1c8d", # VITISAI-remove-unused-code
            "1b081d51dc5c17548bcc04be1f976640f7299333", # VITISAI-node-arg-can-be-used-more-than-once
            "87285323e683e7adc48cf3eab17a00ca0b4fe8e2", # VITISAI-nested-subgraph-is-unsupported-for-now
            "df124c9313a510575a9bf4590704cf10feef0fa6", # 1. Fix reading .dat and .onnx on Linux 2. Fix issue of compiling graph twice
            Path(__file__).parent / ".." / "patch" /
            "onnx_graph_save_supports_subgraph.patch",
        ]
