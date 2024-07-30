from .tools.recipe import *
import logging
from pathlib import Path
from glob import glob
from shutil import copy


class transformers_header(Recipe):

    def __init__(self):
        super().__init__("transformers_header")
    
    def git_url(self):
        return "https://gitenterprise.xilinx.com/VitisAI/transformers.git"
        
    def git_branch(self):
        return "main"
    
    def config(self):
        pass
    
    def build(self):
        pass
    
    def need_install(self):
        return True
    
    def install(self):
        with shell.Cwd(self.src_dir() / "ops" / "cpp"):
            p = Path("*/*.h*")
            files = glob(str(p))
            for file in files:
                dst = self.install_prefix() / "include/transformers/include" / Path(file)
                dst.parent.mkdir(parents=True, exist_ok = True)
                logging.info(f"Copying {file} to {str(dst)}...")
                copy(file, dst)
