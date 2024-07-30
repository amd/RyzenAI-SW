import re
from recipes.tools import *

def parse_release_file(all_recipes, file_name):
    commit_ids = {}
    with open(file_name, "r") as f:
        lines = f.readlines()
        lines = [re.sub("#.*", "", line).strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        for line in lines:
            words = line.split(":")
            words = [word.strip(" \t") for word in words]
            commit_ids[words[0]] = words[1]

    for r in all_recipes:
        if r.name() in commit_ids:
            r.set_git_commit(commit_ids[r.name()])