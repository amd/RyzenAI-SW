from .tools.cmake_recipe import *
from .tools import shell
import os


class protobuf(CMakeRecipe):

    def __init__(self):
        super().__init__('protobuf')

    def git_url(self):
        return "https://github.com/protocolbuffers/protobuf.git"

    def git_branch(self):
        return "v3.21.1"

    def git_commit(self):
        return super().git_commit() or "v3.21.1"

    def CMakeLists_txt_dir(self):
        return "cmake"

    def cmake_extra_args(self):
        args = [
            '-Dprotobuf_MSVC_STATIC_RUNTIME=OFF',
            '-Dprotobuf_BUILD_TESTS=OFF',
        ]

        args.append('-Dprotobuf_BUILD_SHARED_LIBS=OFF')
        args.append('-DBUILD_SHARED_LIBS=OFF')
        if self.is_crosss_compilation():
            args.append('-Dprotobuf_BUILD_PROTOC_BINARIES=OFF')
        return args

    def install_crosscompiling_native_require(self):
        try:
            host_dir = f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr"
            st = os.stat(host_dir)
            mask = oct(st.st_mode)[-3:]
            if st.st_uid != os.getuid() and int(mask) != 777:
                chmod_cmd = [
                    "sudo", "chmod", "777", "-R",
                    f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr"
                ]
                self._run(chmod_cmd)
        except Exception as e:
            logging.warning(
                f"!!! warning :( !!!  change mode 777 {os.environ['OECORE_NATIVE_SYSROOT']} failed!.)"
            )
        try:
            import pip
            import wheel
        except ImportError as e:
            self.install_pip_and_packaging()

        self.install_cmake()
        self.install_gtest()

    def install_gtest(self):
        gtest_dir = "googletest-release-1.12.1"
        tar_ball = "release-1.12.1.tar.gz"
        if not os.path.exists(tar_ball):
            wget_cmd = [
                "wget",
                "https://github.com/google/googletest/archive/refs/tags/" + tar_ball
            ]
            self._run(wget_cmd)
            self._run(["tar", "-xzvf", tar_ball])
        gtest_build_dir = self.build_dir() / "../gtest"
        os.makedirs(gtest_build_dir, exist_ok=True)
        self._run([ 
            "cmake", "-S", gtest_dir, "-B", gtest_build_dir,
        ])

        self._run([ 
            "cmake", "--build", gtest_build_dir
        ])

        install_cmd = []
        host_dir = f"{os.environ['OECORE_TARGET_SYSROOT']}/usr"
        st = os.stat(host_dir)
        if st.st_uid != os.getuid():
            install_cmd.extend(["sudo"])
        install_cmd.extend([ 
            "cmake", "--install", gtest_build_dir, "--prefix", os.environ['OECORE_TARGET_SYSROOT'] + "/usr/"
        ])
        self._run(install_cmd)

    def install_cmake(self):
        cmake_dir = "cmake-3.26.3-linux-x86_64"
        if not os.path.exists("%s.sh" % cmake_dir):
            wget_cmd = [
                "wget",
                f"https://github.com/Kitware/CMake/releases/download/v3.26.3/{cmake_dir}.sh"
            ]
            self._run(wget_cmd)

        install_cmd = []
        host_dir = f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr"
        st = os.stat(host_dir)
        if st.st_uid != os.getuid():
            install_cmd.extend(["sudo"])
        install_cmd.extend([
            "sh", f"{cmake_dir}.sh", "--skip-license",
            f"--prefix={os.environ['OECORE_NATIVE_SYSROOT']}/usr/"
        ])
        self._run(install_cmd)

    def install_pip_and_packaging(self):
        wget_cmd = [
            "wget", "-P", "/tmp", "https://bootstrap.pypa.io/get-pip.py"
        ]
        install_cmd = ["python", "/tmp/get-pip.py"]
        packaging_cmd = ["python", "-m", "pip", "install", "packaging"]
        if not os.path.exists("/tmp/get-pip.py"):
            self._run(wget_cmd)
        self._run(install_cmd)
        self._run(packaging_cmd)

    def config(self):
        import multiprocessing
        num_parallel_jobs = multiprocessing.cpu_count()
        if self.is_crosss_compilation():
            self.install_crosscompiling_native_require()

            ## overwrite the install tools in SDK.
            os.makedirs(self.build_dir() / "native", exist_ok=True)
            with shell.Cwd(self.build_dir() / "native"):
                self._run([
                    "env",
                    "-i",
                    "PATH=/usr/bin:/usr/local/bin",
                    f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr/bin/cmake",
                    "-S",
                    self.src_dir() / "cmake",
                    "-B",
                    self.build_dir() / "native",
                    "-Dprotobuf_BUILD_TESTS=off",
                    "-DBUILD_SHARED_LIBS=off",
                    #"-DCMAKE_INSTALL_RPATH=\"\"",
                    "-DCMAKE_SKIP_INSTALL_RPATH=TRUE",
                ])
                self._run([
                    "env",
                    "-i",
                    "PATH=/usr/bin:/usr/local/bin",
                    f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr/bin/cmake",
                    "--build",
                    self.build_dir() / "native",
                    "-j",
                    str(num_parallel_jobs),
                ])
                host_dir = f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr"
                st = os.stat(host_dir)
                install_cmd = []
                if st.st_uid != os.getuid():
                    install_cmd.extend(["sudo"])
                install_cmd.extend([
                    "env",
                    "-i",
                    "PATH=/usr/bin:/usr/local/bin",
                    f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr/bin/cmake",
                    "--install",
                    self.build_dir() / "native",
                    "--prefix",
                    f"{os.environ['OECORE_NATIVE_SYSROOT']}/usr",
                ])
                self._run(install_cmd)
        return super().config()
