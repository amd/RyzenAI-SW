from .tools.recipe import *
import urllib.request
import zipfile
import logging


class tvm_aie_compiler_artifact(Recipe):

    def __init__(self):
        super().__init__('tvm_aie_compiler_artifact')
    
    def git_url(self):
        return "https://gitenterprise.xilinx.com/VitisAI/tvm_aie_compiler_artifact"
        
    def git_branch(self):
        return "dev"
    
    def config(self):
        pass
    
    def build(self):
        pass
    
    def need_install(self):
        return True
    
    def _delete_tags(self):
        from subprocess import run
        ret = run(["git", "tag", "-l"], capture_output = True)
        for tag in ret.stdout.decode().strip().split('\n'):
            if tag != '':
                self._run(["git", "tag", "-d", tag])
    
    def checkoutByCommit(self):
        self._delete_tags()
        self._run(["git", "fetch", "--all", "--tags"])
        match_obj = re.match('pr(\d+)', self.git_commit())
        if match_obj:
            self._run([
                "git", "fetch", "-u", "origin",
                f"pull/{match_obj.group(1)}/head:{self.git_commit()}"
            ])
        self._run(["git", "checkout", "--force", self.git_branch()])
        self._run(["git", "pull", "--rebase"])
        try:
            self._run(["git", "tag", self.git_commit()])
        except:
            self._run(["git", "checkout", "--force", self.git_commit()])

    def install(self):
        with shell.Cwd(self.src_dir()):
            self._run(['python', 'install.py',
                '--download-path', str(self.build_dir()),
                '--install-path', str(self.install_prefix())
            ])
