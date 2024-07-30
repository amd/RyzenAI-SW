import argparse
import os
import zipfile
import urllib.request
import platform
import subprocess
import re
import sys
from datetime import datetime
from pathlib import Path
from recipes.tools.shell import Cwd
import shutil
import hashlib
import ssl

TMP_DIR = Path.home() / "build" / "tmp"


def copy_to_zip_imp(zip, src, dst):
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for file in files:
                new_src = os.path.join(root, file)
                new_dst = dst / os.path.relpath(os.path.join(root, file), src)
                zip.write(new_src, new_dst)
    else:
        zip.write(src, dst)


def add_onnxruntime_manifest(meta, recipe):
    return meta + recipe.get_package_content()


def add_manifest(meta, prefix, recipe_name):
    ret = []
    with open(Path.cwd() / "recipes" /
              (recipe_name + "_install_manifest.txt")) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            ret.append((prefix + "/" + line, Path(recipe_name) / line))
    return meta + ret


def copy_to_zip(meta, name):
    path = TMP_DIR.parent / name
    print(path)
    if os.path.isfile(path):
        os.remove(path)

    compression = zipfile.ZIP_DEFLATED
    compresslevel = 9

    try:
        with zipfile.ZipFile(path,
                             'w',
                             compression=compression,
                             compresslevel=compresslevel) as zip:
            for m in meta:
                copy_to_zip_imp(zip, m[0], m[1])
    except FileNotFoundError as e:
        print(f"Packing failed: {str(e)}")
        if os.path.isfile(path):
            os.remove(path)


def get_meta(recipe_dict):
    vaip_recipe = recipe_dict["vaip"]
    test_onnx_runner_recipe = recipe_dict["test_onnx_runner"]
    onnx_recipe = recipe_dict["onnxruntime"]
    install_prefix = vaip_recipe.install_prefix()
    bin = install_prefix / "bin"
    test_case_src = test_onnx_runner_recipe.src_dir()
    vaip_build_dir = vaip_recipe.build_dir()
    onnx_build_dir = onnx_recipe.build_dir() / onnx_recipe.build_type(
    ) / onnx_recipe.build_type()
    pkg_src = Path(os.path.dirname(os.path.realpath(__file__))) / "package_src"
    return [
        ## resnet50 cxx sample
        (pkg_src, Path("/")),
        (test_case_src / "word_list.inc",
         Path("/") / "vitis_ai_ep_cxx_sample" / "resnet50" / "word_list.inc"),
        (test_case_src / "util",
         Path("/") / "vitis_ai_ep_cxx_sample" / "resnet50" / "util"),
        (test_case_src / "data" / "resnet50.jpg",
         Path("/") / "vitis_ai_ep_cxx_sample" / "resnet50" / "resnet50.jpg"),
        (test_case_src / "resnet50_pt.cpp", Path("/") /
         "vitis_ai_ep_cxx_sample" / "resnet50" / "resnet50_pt.cpp"),
        (onnx_build_dir / "onnxruntime_perf_test.exe",
         Path("/") / "bin" / "onnxruntime_perf_test.exe"),
        ## resnet50 python sample
        (test_case_src / "resnet50_python",
         Path("/") / "vitis_ai_ep_py_sample" / "resnet50_python"),
        (test_case_src / "win_scripts_2", Path("/") / "vitis_ai_ep_py_sample"),
        ## models
        (test_case_src / "data" / "pt_resnet50.onnx",
         Path("/") / "models" / "resnet50_pt.onnx"),
        (test_case_src / "data" / "pt_resnet50_test_data_set_0",
         Path("/") / "models" / "test_data_set_0"),
        (bin / "vaip_config.json", Path("/") / "vaip_config.json"),
        ## python whl files
        (vaip_build_dir / "onnxruntime_vitisai_ep" / "python" / "dist",
         Path("/")),
        (onnx_build_dir / "dist", Path("/")),
        ##
        (bin / "onnxruntime_vitisai_ep.dll",
         Path("/") / "bin" / "onnxruntime_vitisai_ep.dll"),
        (bin / "vitis-ai-runtime.dll",
         Path("/") / "bin" / "vitis-ai-runtime.dll"),
        (bin / "resnet50_pt.exe", Path("/") / "bin" / "resnet50_pt.exe"),
        (bin / "test_dpu_subgraph.exe",
         Path("/") / "bin" / "test_dpu_subgraph.exe"),
        (bin / "test_onnx_runner.exe",
         Path("/") / "bin" / "test_onnx_runner.exe"),
        (bin / "test_xmodel.exe",
         Path("/") / "bin" / "test_xmodel.exe"),
        (bin / "classification.exe",
         Path("/") / "bin" / "classification.exe"),
        (TMP_DIR / "version_info.txt", Path("/") / "version_info.txt"),
    ]


def install_python_package(python_path, recipe_dict):
    url = "https://bootstrap.pypa.io/get-pip.py"
    downloadPath = TMP_DIR / "get-pip.py"
    if os.path.exists(downloadPath) == False:
        urllib.request.urlretrieve(url, downloadPath)
    import subprocess
    subprocess.call([python_path, TMP_DIR / "get-pip.py"])
    onnxruntime = recipe_dict["onnxruntime"]
    onnxruntime_whl = list(
        (onnxruntime.build_dir() / onnxruntime.build_type() /
         onnxruntime.build_type() / "dist").glob('onnxruntime*.whl'))[0]
    voe_whl = list(
        (recipe_dict["vaip"].build_dir() / "onnxruntime_vitisai_ep" /
         "python" / "dist").glob('voe*.whl'))[0]
    subprocess.call([
        python_path, "-m", "pip", "install", voe_whl, onnxruntime_whl,
        "imageio", "--force-reinstall", "--no-warn-script-location"
    ])


def add_python(meta, recipe_dict):
    python_version = sys.version
    match = re.search(r'(\d+\.\d+\.\d+)', python_version)
    version_number = "3.9.7"
    if match:
        version_number = match.group(1)
    python_embed = "python-" + version_number + "-embed-amd64.zip"
    url = "https://www.python.org/ftp/python/" + version_number + "/" + python_embed
    downloadPath = TMP_DIR / python_embed
    ssl._create_default_https_context = ssl._create_unverified_context
    if os.path.exists(downloadPath) == False:
        urllib.request.urlretrieve(url, downloadPath)
    dstPath = TMP_DIR / "python"
    if os.path.exists(dstPath) == True:
        shutil.rmtree(dstPath)
    with zipfile.ZipFile(downloadPath, 'r') as embed:
        embed.extractall(dstPath)
    major, minor, _ = version_number.split(".")
    with open(dstPath / ("python" + major + minor + "._pth"), "a") as pth:
        pth.write("lib\n")
        pth.write("dlls\n")
        pth.write("lib\\site-packages\n")
        pth.write("..\\xrt \n")
    install_python_package(dstPath / "python", recipe_dict)
    meta.append((dstPath, Path("/") / "python"))
    return meta


def calculate_file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def add_xclbin(meta, note):
    for s in note:
        if s.startswith('xclbin_url:'):
            url = s.replace('xclbin_url:', '')
            xclbin_name = url[url.rindex("/") + 1:]
            downloadPath = TMP_DIR / xclbin_name
            if os.path.isfile(downloadPath):
                os.remove(downloadPath)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            for i in range(10):
                urllib.request.urlretrieve(url, downloadPath)
                if os.path.isfile(downloadPath
                                  ) and os.stat(downloadPath).st_size > 102400:
                    break
                else:
                    if i == 9:
                        exit(-1)
                    print(
                        f'Maybe the download of xclbin failed, the size of xclbin is { os.stat(downloadPath).st_size /1024}KB, try to download again for the {i+1} time'
                    )
            print(downloadPath, calculate_file_md5(downloadPath))
            meta.append((downloadPath, Path("/") / xclbin_name))
    return meta


def generate_version_info_file(all_recipe, note):
    versionInfoBuilder = ""
    #add vai-rt version
    with Cwd(os.path.dirname(os.path.realpath(__file__))):
        commit_id = subprocess.check_output(["git", "rev-parse",
                                             "HEAD"]).decode('ascii').strip()
        versionInfoBuilder = versionInfoBuilder + f"#vai-rt: {commit_id}\n"

    for s in note:
        if s.startswith("xclbin_url"):
            versionInfoBuilder += f"#{s}\n"

    for r in all_recipe:
        commit_id = ""
        if r.git_commit() != None:
            commit_id = r.git_commit()
        else:
            commit_id = r.git_current_commit_id()
        versionInfoBuilder = versionInfoBuilder + f"{r.name()}: {commit_id}\n"

    filename = "version_info.txt"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / filename
    with open(path, "w") as f:
        f.write(versionInfoBuilder)


def create_package(arg_setting, all_recipe, note, file_path):
    if platform.system() != "Windows":
        return
    if arg_setting.package == False:
        return
    recipe_dict = {}
    for r in all_recipe:
        recipe_dict[r.name()] = r
    global TMP_DIR
    TMP_DIR = recipe_dict["vaip"].build_dir().parent / "tmp"
    print(f'tmp dir: {TMP_DIR}')
    generate_version_info_file(all_recipe, note)

    meta = get_meta(recipe_dict)
    meta = add_python(meta, recipe_dict)
    meta = add_xclbin(meta, note)
    meta = add_onnxruntime_manifest(meta, recipe_dict["onnxruntime"])
    """
    # Use this code when you need to package opencv and eigen
    prefix = str(recipe_dict["vaip"].install_prefix())
    meta = add_manifest(meta, prefix, "opencv")
    meta = add_manifest(meta, prefix, "eigen")
    """
    if file_path:
        suffix = os.path.splitext(os.path.basename(file_path))[0]
    else:
        suffix = "None"
    with Cwd(os.path.dirname(os.path.realpath(__file__))):
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode('ascii').strip()
        suffix = commit_id + "-" + suffix
    name = f"voe-win_amd64-with_xcompiler_{os.environ.get('WITH_XCOMPILER', 'ON').lower()}-{suffix}.zip"

    copy_to_zip(meta, name)
