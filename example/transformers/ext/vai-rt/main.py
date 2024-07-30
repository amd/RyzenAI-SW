from pathlib import Path
import traceback
import sys
import os
import platform
import logging

HOME = Path.home()


def setup_envs(exists=True):
    import setuptools.msvc
    msvc_env = setuptools.msvc.EnvironmentInfo('x64')
    env = dict(
        include=msvc_env._build_paths('include', [
            msvc_env.VCIncludes, msvc_env.OSIncludes, msvc_env.UCRTIncludes,
            msvc_env.NetFxSDKIncludes
        ], exists),
        lib=msvc_env._build_paths('lib', [
            msvc_env.VCLibraries, msvc_env.OSLibraries, msvc_env.FxTools,
            msvc_env.UCRTLibraries, msvc_env.NetFxSDKLibraries
        ], exists),
        libpath=msvc_env._build_paths('libpath', [
            msvc_env.VCLibraries,
            msvc_env.FxTools,
            msvc_env.VCStoreRefs,
            msvc_env.OSLibpath,
            msvc_env.MSBuild,
        ], exists),
        path=msvc_env._build_paths('path', [
            msvc_env.VCTools, msvc_env.VSTools, msvc_env.VsTDb,
            msvc_env.SdkTools, msvc_env.SdkSetup, msvc_env.FxTools,
            msvc_env.MSBuild, msvc_env.HTMLHelpWorkshop, msvc_env.FSharp
        ], exists),
    )
    logging.info(f"{env}")
    os.environ.update(env)
    build_env = HOME / "build.env"
    your_env = {}
    if os.path.isfile(build_env):
        with open(build_env, encoding='utf-8') as f:
            your_env = {
                k: v.rstrip()
                for s in f.readlines() if not s.startswith('#')
                for [k, v] in [s.split("=", 2)]
            }
    os.environ.update(your_env)
    logging.info(f"{your_env}")
    return


def main(args):
    logging.basicConfig(
        format=
        '%(levelname)s [%(module)s::%(funcName)s]:%(lineno)d %(message)s',
        level=logging.DEBUG)
    tb = None
    if platform.system() == "Windows":
        setup_envs()
    try:
        import run
        run.main(args)
    except Exception:
        tb = traceback.format_exc()
    finally:
        if not tb is None:
            print(tb)
            sys.exit(1);    

if __name__ == "__main__":
    main(sys.argv[1:])
