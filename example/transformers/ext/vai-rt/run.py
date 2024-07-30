import logging
import os
import sys
import argparse
import platform
from pathlib import Path
import recipes
from recipes.tools import cmake_recipe
from recipes.tools import shell
from package import create_package
from release_file import parse_release_file

IS_CROSS_COMPILATION = False
IS_WINDOWS = False
IS_NATIVE_COMPILATION = False
if "OECORE_TARGET_SYSROOT" in os.environ:
    IS_CROSS_COMPILATION = True
elif platform.system() == "Windows":
    IS_WINDOWS = True
else:
    IS_NATIVE_COMPILATION = True


def load_json_recipe(recipe_file):
    import json

    recipe_def = json.loads(open(recipe_file).read())
    return [
        cmake_recipe.load(recipe_def, proj) for proj in recipe_def.get("projects", [])
    ]


# This function determines the required recipes from the release file
# It also sets the users desired commit ids in the recipe
import re


def load_release_recipe(release_file):
    commit_ids = {}
    with open(release_file, "r") as f:
        lines = f.readlines()
        lines = [re.sub("#.*", "", line).strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        for line in lines:
            words = line.split(":")
            words = [word.strip(" \t") for word in words]
            commit_ids[words[0]] = words[1]
    valid_recipes = get_all_recipes()
    valid_recipe_names_set = set([r.name() for r in valid_recipes])
    all_recipes = []

    for key in commit_ids:
        k = key.replace("-", "_")
        r = recipes.__dict__[k].__dict__[k]()
        if r.name() in valid_recipe_names_set:
            all_recipes.append(r)

    for r in all_recipes:
        if r.name() in commit_ids:
            r.set_git_commit(commit_ids[r.name()])
            if r.name() == 'graph-engine':
                os.environ["GRAPH_ENGINE_COMMIT_ID"] = commit_ids[r.name()]
    return all_recipes


def load_release_note(release_file):
    with open(release_file, "r") as f:
        lines = f.readlines()
        return [s[1:].replace("\n", "") for s in lines if s.startswith("#")]
    return []


def get_all_recipes():
    return [
        r
        for r in [
            recipes.boost.boost() if IS_NATIVE_COMPILATION or IS_WINDOWS else None,
            recipes.glog.glog(),
            recipes.gsl.gsl(),
            recipes.protobuf.protobuf()
            if (IS_CROSS_COMPILATION and os.environ["OECORE_SDK_VERSION"] != "2023.1")
            or not IS_CROSS_COMPILATION
            else None,
            recipes.eigen.eigen() if IS_NATIVE_COMPILATION or IS_WINDOWS else None,
            recipes.pybind11.pybind11()
            if IS_NATIVE_COMPILATION or IS_WINDOWS
            else None,
            recipes.opencv.opencv() if IS_WINDOWS else None,
            recipes.rapidjson.rapidjson() if IS_NATIVE_COMPILATION else None,
            recipes.gtest.gtest() if IS_WINDOWS else None,
            recipes.xrt.xrt() if IS_NATIVE_COMPILATION else None,
            recipes.unilog.unilog(),
            recipes.xir.xir(),
            recipes.target_factory.target_factory(),
            recipes.trace_logging.trace_logging() if IS_WINDOWS else None,
            recipes.vart.vart(),
            recipes.xcompiler.xcompiler()
            if os.environ.get("WITH_XCOMPILER", "ON") == "ON"
            else None,
            recipes.graph_engine.graph_engine() if IS_WINDOWS else None,
            recipes.vairt.vairt() if IS_WINDOWS else None,
            recipes.testcases.testcases() if IS_WINDOWS else None,
            recipes.onnxruntime.onnxruntime(),
            recipes.zlib.zlib() if IS_WINDOWS else None,
            recipes.spdlog.spdlog() if IS_WINDOWS else None,
            recipes.tvm_aie_compiler_artifact.tvm_aie_compiler_artifact()
            if os.environ.get("WITH_TVM_AIE_COMPILER", "OFF") == "ON" and IS_WINDOWS else None,
            recipes.transformers_header.transformers_header()
            if os.environ.get("WITH_TVM_AIE_COMPILER", "OFF") == "ON" and IS_WINDOWS else None,
            recipes.vaip.vaip(),
            recipes.test_onnx_runner.test_onnx_runner(),
        ]
        if not r is None
    ]


def main(args):
    parser = argparse.ArgumentParser(description="A simple helper for cmake.")
    parser.add_argument(
        "--clean", action="store_true", help="discard build dir before build"
    )
    parser.add_argument(
        "--release_file",
        type=Path,
        help="set all internal repos to commits from file, use 'latest' to use latest release",
    )
    parser.add_argument("--dry-run", action="store_true", help="do not run any command")
    parser.add_argument(
        "--update-src", action="store_true", help="update source directory"
    )
    parser.add_argument(
        "--dev-mode", action="store_true", help="enable dev mode or not"
    )
    # quoted, exmaple: --foreach "ls -lh" "echo hello"
    parser.add_argument(
        "--foreach",
        nargs="+",
        default=[],
        help="run the following command in each own project",
    )
    parser.add_argument(
        "--type", choices=["debug", "release", "Debug", "Release"], default=None
    )
    parser.add_argument(
        "--project", nargs="+", default=[], help="build specific projects"
    )
    parser.add_argument(
        "--exclude-project", nargs="+", default=[], help="exluding specific projects"
    )
    parser.add_argument("--json-recipes", help="a json file to specify all recipies")
    parser.add_argument(
        "--package", action="store_true", help="if you want to create a package"
    )
    arg_setting, unknown_args = parser.parse_known_args(args)
    if len(unknown_args) != 0:
        logging.error(f"unknown arguments: {unknown_args}")
        return
    if arg_setting.json_recipes:
        all_recipes = load_json_recipe(arg_setting.json_recipes)
    elif arg_setting.release_file:
        all_recipes = load_release_recipe(arg_setting.release_file)
    else:
        all_recipes = get_all_recipes()
    logging.basicConfig(
        format="%(levelname)s [%(module)s::%(funcName)s]:%(lineno)d %(message)s",
        level=logging.DEBUG,
    )

    if arg_setting.dry_run:
        for recipe in all_recipes:
            recipe.dry_run = True

    if arg_setting.update_src:
        for recipe in all_recipes:
            recipe.update_src_dir = True

    if arg_setting.project:
        all_recipes = [r for r in all_recipes if r.name() in arg_setting.project]

    if arg_setting.exclude_project:
        all_recipes = [
            r for r in all_recipes if not r.name() in arg_setting.exclude_project
        ]

    ## todo, command args should have higher priority.
    if arg_setting.dev_mode:
        os.environ["DEV_MODE"] = "1"
    if not arg_setting.type is None:
        os.environ["BUILD_TYPE"] = arg_setting.type.capitalize()

    if len(arg_setting.foreach) > 0:
        for r in all_recipes:
            logging.info(f"running command for {r.name()}")
            with shell.Cwd(r.src_dir()):
                for cmd in arg_setting.foreach:
                    r.run(cmd.split(" "))
        sys.exit(0)

    if arg_setting.clean:
        for recipe in all_recipes:
            recipe.clean()

    for recipe in all_recipes:
        recipe.make_all()

    note = []
    if arg_setting.release_file:
        note = load_release_note(arg_setting.release_file)

    create_package(arg_setting, all_recipes, note, arg_setting.release_file)
