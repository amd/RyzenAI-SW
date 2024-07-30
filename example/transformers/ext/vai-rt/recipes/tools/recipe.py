from pathlib import Path
import re
import os
import sys
import logging
import multiprocessing
from . import workspace
from . import shell
import platform
import shutil
import time
import subprocess

HOME = Path.home()

if platform.system() == "Windows":
    SYSTEM = platform.system()
    VERSION_ID = platform.version()
elif os.path.isfile("/etc/lsb-release"):
    with open("/etc/lsb-release", encoding='utf-8') as f:
        lsb_info = {
            k: v.rstrip()
            for s in f.readlines()
            for [k, v] in [s.split("=", 2)]
        }
        SYSTEM = lsb_info['DISTRIB_ID']
        VERSION_ID = lsb_info['DISTRIB_RELEASE']
else:
    SYSTEM = platform.system()
    VERSION_ID = platform.version()

def is_valid_commit_id(commit_id):
    pattern = re.compile(r'^[0-9a-fA-F]{40}$')
    return bool(pattern.match(commit_id))

class Recipe(workspace.Workspace):

    def __init__(self, name):
        super().__init__()
        self._name = name
        self._udpate_src_dir = False
        self._commit_id = None
        self._prefer_https_git_url = os.environ.get('PREFER_HTTPS_GIT_URL',
                                                    False)

    def name(self):
        return self._name

    def git_url(self):
        return "https://gitlab/" + self.name()

    def git_branch(self):
        return "master"

    def set_git_commit(self, commit_id):
        self._commit_id = commit_id

    def git_commit(self):
        return self._commit_id

    def git_current_commit_id(self):
        #may fail if it is not from a git repo
        with shell.Cwd(self.src_dir()):
            return subprocess.check_output(["git", "rev-parse",
                                            "HEAD"]).decode('ascii').strip()

    def get_package_content(self):
        #onnxrutime is not a cmake recipe, so it is implemented in recipe class
        ret = []
        if hasattr(self, "manifest_dir"):
            with open(self.manifest_dir() / "install_manifest.txt") as f:
                for line in f.readlines():
                    line = line.replace('\n', '')
                    ret.append(
                        (line, os.path.relpath(line, self.install_prefix())))
        return ret

    def need_for_package(self):
        return False

    def git_patch(self):
        return None

    def build_type(self):
        return os.environ.get('BUILD_TYPE', "Debug")

    def build_dir(self):
        if 'VAI_RT_BUILD_DIR' in os.environ:
            return Path(os.environ['VAI_RT_BUILD_DIR']) / self.name()
        elif int(os.environ.get('DEV_MODE', '0')) == 1:
            return HOME / "build" / ("build." +
                                     self.target_info()) / self.name()
        else:
            return HOME / "build" / ("build.external." +
                                     self.target_info()) / self.name()

    def src_dir(self):
        return self.workspace() / self.name()

    @property
    def update_src_dir(self):
        return self._udpate_src_dir

    @update_src_dir.setter
    def update_src_dir(self, value):
        self._udpate_src_dir = value

    def build_static(self):
        return True

    def apply_patch(self):
        patch_list = self.git_patch()
        for p in patch_list:
            patch_file_path = None
            is_commit_id = is_valid_commit_id(str(p))
            # if patch is a commit id, use it to generate a patch file
            if is_commit_id:
                format_output = subprocess.run(["git", "format-patch", '-1', p], check=True, capture_output=True)
                patch_file_path = format_output.stdout.decode().split("\n")[0]
            else:
                patch_file_path = p

            self._run(["git", "apply", patch_file_path])
            # delete generated patch after apply
            if is_commit_id:
                os.remove(patch_file_path)

    def checkoutByCommit(self):
        match_obj = re.match('pr(\d+)', self.git_commit())
        if match_obj:
            self._run([
                "git", "fetch", "-u", "origin",
                f"pull/{match_obj.group(1)}/head:{self.git_commit()}"
            ])
        self._run(["git", "checkout", "--force", self.git_commit()])

    def update_to_latest(self):
        self._run(["git", "fetch", "--all"])
        if self.git_commit():
            self.checkoutByCommit()
        else:
            self._run(["git", "checkout", "--force", self.git_branch()])
            self._run(["git", "pull", "--rebase"])
        self._run(["git", "clean", "-xfd"])
        self._run(["git", "checkout", "."])
        self._run(["git", "rev-parse", "HEAD"])
        self.apply_patch()

    def download(self):
        if not os.path.isdir(self.src_dir()):
            with shell.Cwd(self.workspace()):
                self._run([
                    "git", "clone",
                    self.git_url(), "--branch",
                    self.git_branch(),
                    self.src_dir()
                ])
                with shell.Cwd(self.src_dir()):
                    self._run(["git", "rev-parse", "HEAD"])
            if self.git_commit():
                with shell.Cwd(self.src_dir()):
                    self.checkoutByCommit()
            with shell.Cwd(self.src_dir()):
                self.apply_patch()
            return

        with shell.Cwd(self.src_dir()):
            self._run(["git", "remote", "-v", "show"])
            if int(os.environ.get('DEV_MODE',
                                  '0')) != 1 or self.update_src_dir:
                self.update_to_latest()

    def workspace(self):
        if 'VAI_RT_WORKSPACE' in os.environ:
            return Path(os.environ['VAI_RT_WORKSPACE'])
        elif int(os.environ.get('DEV_MODE', '0')) == 1:
            return self._dev_workspace()
        else:
            return HOME / "build" / "external_workspace"

    def _dev_workspace(self):
        if 'WORKSPACE' in os.environ:
            return Path(os.environ['WORKSPACE'])
        if self.is_windows():
            return HOME / "workspace"
        else:
            return Path("/") / "workspace"

    def config(self):
        logging.warn('not implemented')

    def build(self):
        logging.warn('not implemented')

    def install(self):
        logging.warn('not implemented')

    def arch(self):
        if self.is_crosss_compilation():
            return os.environ['OECORE_TARGET_ARCH']
        else:
            return platform.machine()

    def system(self):
        if self.is_crosss_compilation():
            return os.environ['OECORE_TARGET_OS']
        else:
            return SYSTEM

    def version_id(self):
        if self.is_crosss_compilation():
            return os.environ['OECORE_SDK_VERSION']
        else:
            return VERSION_ID

    def git_patch(self):
        return []

    def target_info(self):
        return ".".join([
            self.system(),
            self.version_id(),
            self.arch(),
            self.build_type().capitalize()
        ])

    def install_prefix(self):
        if "VAI_RT_PREFIX" in os.environ:
            return Path(os.environ.get("VAI_RT_PREFIX"))
        elif self.is_crosss_compilation():
            return Path(os.environ['OECORE_TARGET_SYSROOT']
                        ) / "install" / self.build_type()
        else:
            return self.home() / ".local" / self.target_info()

    def clean(self):
        logging.info(f"removing build directory: {self.build_dir()}")
        if not self.dry_run:
            shutil.rmtree(self.build_dir(), ignore_errors=True)

    def need_install(self):
        return True

    def report(self):
        logging.info(f"=== begin ===  start to build [{self.name()}]")
        logging.info(f"\tworkspace={self.workspace()}")
        logging.info(f"\tlog_file={self.log_file()}")
        logging.info(f"\tsrc_dir={self.src_dir()}")
        logging.info(f"\tbuild_type={self.build_type()}")
        logging.info(f"\tbuild_dir={self.build_dir()}")
        logging.info(f"\tinstall_dir={self.install_prefix()}")
        # logging.info(f"\tneed_download={self.need_download()}")
        # logging.info(f"\tneed_config={self.need_config()}")
        # logging.info(f"\tneed_build={self.need_build()}")
        # logging.info(f"\tneed_install={self.need_install()}")

    def make_all(self):
        try:
            self.report()
            start = time.time()
            self._clean_log_file()
            logging.info(f"\tstart to download {self.name()}")
            self.download()

            logging.info(f"\tstart to config {self.name()}")
            self.config()

            logging.info(f"\tstart to build {self.name()}")
            self.build()

            if self.need_install():
                logging.info(f"\tstart to install {self.name()}")
                self.install()
            else:
                logging.info(f"\t{self.name()} is already installed")
        except subprocess.CalledProcessError as e:
            logging.error(
                f"!!! failure :( !!!  build [{self.name()}] failed!. cmd= {e.cmd} log {self.log_file()} are show as below:"
            )
            self._show_log_file()
            sys.exit(1)
        else:
            end = time.time()
            elapse = "{:.2f}".format(end - start)
            logging.info(
                f"=== end :) ==== build [{self.name()}] {elapse} seconds done.\n\tplease read {self.log_file()} for details"
            )
        finally:
            pass
