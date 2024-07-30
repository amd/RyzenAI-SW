#!/usr/bin/env python
import multiprocessing
import subprocess
import argparse
import shutil
import os
import platform
import shlex
from pathlib import Path

home = Path.home()


def bool_arg(s):
    yes_table = {'yes', 'on', 'true'}
    return s.lower() in yes_table


def run_shell(args, dry_run):
    print("running: " + " ".join([shlex.quote(arg) for arg in args]))
    if not dry_run:
        subprocess.check_call(args)


def main(args):
    parser = argparse.ArgumentParser(description='A simple helper for cmake.')
    parser.add_argument("--clean",
                        action='store_true',
                        help="discard build dir before build")
    parser.add_argument("--verbose",
                        action='store_true',
                        help="discard build dir before build")
    parser.add_argument("--dry-run",
                        action='store_true',
                        help="dont not run cmake actually")
    parser.add_argument("--static_lib",
                        action='store_true',
                        help="enable shared lib build or not")
    parser.add_argument("--src-dir",
                        type=Path,
                        default=Path(os.getcwd()),
                        help="root directory of source code")
    parser.add_argument("--use-ninja",
                        help="use ninja generator or not",
                        type=bool_arg,
                        default="True" if shutil.which("ninja") else "False")

    parser.add_argument("--build-python", action='store_true')
    parser.add_argument("--type",
                        choices=['debug', 'release'],
                        default='debug')

    # if platform.system
    arch = platform.machine()
    system = platform.system()
    distributor_id = "MS"
    version_id = platform.version()
    if 'OECORE_TARGET_SYSROOT' in os.environ:
        system = "linux"
        version_id = os.environ['OECORE_SDK_VERSION']
        arch = os.environ['OECORE_TARGET_ARCH']
    elif platform.system() == "Linux":
        with open("/etc/lsb-release", encoding='utf-8') as f:
            lsb_info = {
                k: v.rstrip()
                for s in f.readlines() for [k, v] in [s.split("=", 2)]
            }
            system = lsb_info['DISTRIB_ID']
            version_id = lsb_info['DISTRIB_RELEASE']

    build_dir_default = home / "build" / ".".join(
        ["build", system, version_id, arch])

    parser.add_argument(
        "--build-dir",
        type=Path,
        help=
        f"set customized build directory. default directory is {build_dir_default}/<BuildType>/<project-name>"
    )
    parser.add_argument("--project-name",
                        default=Path(os.getcwd()).name,
                        help="project name")
    parser.add_argument("--cmake-dir",
                        type=Path,
                        default=Path("."),
                        help="where CMakeLists.txt exits relative to src_dir")

    install_prefix_default = home / ".local" / ".".join(
        [system, version_id, arch])

    parser.add_argument(
        "--install-prefix",
        type=Path,
        help=
        f"set customized install prefix. default prefix is f{install_prefix_default}"
    )

    parser.add_argument(
        "--prefix-path",
        type=Path,
        help=f"prefix. default prefix is f{install_prefix_default}")

    parser.add_argument("--cmake-options",
                        help="append more cmake options",
                        action='append')

    settings = parser.parse_args(args)
    settings.type = settings.type.capitalize()

    if settings.build_dir == None:
        settings.build_dir = build_dir_default = home / "build" / ".".join([
            "build", system, version_id, arch, settings.type
        ]) / settings.project_name

    if settings.install_prefix == None:
        settings.install_prefix = home / ".local" / ".".join(
            [system, version_id, arch, settings.type])
        if platform.system() == "Windows":
            settings.install_prefix = settings.install_prefix / settings.project_name

    if settings.prefix_path == None:
        settings.prefix_path = home / ".local" / ".".join(
            [system, version_id, arch, settings.type])

    args = ["cmake"]

    if not settings.static_lib:
        args.extend(['-DBUILD_SHARED_LIBS=ON'])

    if platform.system() == "Windows":
        args.extend(['-DCMAKE_CONFIGURATION_TYPES=' + settings.type])
    else:
        args.extend(['-DCMAKE_BUILD_TYPE=' + settings.type])

    args.extend(['-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
    args.extend(['-S', str(settings.src_dir / settings.cmake_dir)])
    args.extend(['-B', str(settings.build_dir)])
    if settings.use_ninja:
        args.extend(['-G', 'Ninja'])
        args.extend(['-DCMAKE_INSTALL_PREFIX=' + str(settings.install_prefix)])
        args.extend(['-DCMAKE_PREFIX_PATH=' + str(settings.prefix_path)])
        args.extend(settings.cmake_options if settings.cmake_options else [])
    if settings.clean:
        if not settings.dry_run:
            shutil.rmtree(settings.build_dir)

    run_shell(args, settings.dry_run)
    run_args = [
        "cmake",
        "--build",
        str(settings.build_dir),
        "--config",
        settings.type,
        "-j",
        str(multiprocessing.cpu_count()),
    ]

    if settings.verbose:
        run_args.append("--verbose")

    run_shell(run_args, settings.dry_run)

    run_shell([
        "cmake",
        "--install",
        str(settings.build_dir),
        "--config",
        settings.type,
    ], settings.dry_run)


if __name__ == "__main__":
    main(sys.args)
