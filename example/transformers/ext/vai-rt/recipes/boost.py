from .tools.recipe import *
import urllib.request
import zipfile
import logging


class boost(Recipe):

    def __init__(self):
        super().__init__('boost')

    def download(self):
        zip_file = self.workspace() / "boost_1_79_0.zip"
        unzip_dir = self.workspace() / "boost"
        if not self.is_file(zip_file):
            logging.info(f"downloading {zip_file}")
            urllib.request.urlretrieve(
                "https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.zip",
                zip_file)
        else:
            logging.info(f"{zip_file} is already downloaded")

        if not self.is_dir(unzip_dir):
            logging.info(f"unziping {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
        else:
            logging.info(f"{zip_file} is already unzipped")

    def src_dir(self):
        return self.workspace() / "boost" / "boost_1_79_0"

    def build(self):
        with shell.Cwd(self.src_dir()):
            self._run(['bootstrap.bat'])
            self._run([
                "b2",
                "install",
                "--prefix=" + str(self.install_prefix() / "boost"),
                "--build-type=complete",
                "address-model=64",
                "architecture=x86",
                "link=static",
                "threading=multi",
                "--with-filesystem",
                "--with-program_options",
                "--with-system",
            ],
                      dry_run=False)

    def git_commit(self):
        # only mark the boost as an external project.
        return super().git_commit() or "v1.79.0"

    def make_all(self):
        if not self.is_windows():
            return
        if self.is_dir(self.install_prefix() / "boost" / "include" /
                       "boost-1_79"):
            return
        try:
            self._clean_log_file()
            self.download()
            self.build()
            # self.install()
        finally:
            self._show_log_file()
