import os
import shutil
from git import Repo

# --- Configuration ---
REPOS = [
    {
        "url": "https://github.com/amd/ryzen-ai-documentation.git",
        "branch": "main",
        "clone_dir": "ryzen_ai_docs_repo",
        "rst_source": "docs"
    },
    {
        "url": "https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark.git",
        "branch": "main",
        "clone_dir": "quark_docs_repo",
        "rst_source": "docs/source"
    },
    {
        "url": "https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark.git",
        "branch": "main",
        "clone_dir": "quark_docs_repo_onnx",
        "rst_source": "docs/source/onnx"
    },
    {
        "url": "https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark.git",
        "branch": "main",
        "clone_dir": "quark_docs_repo_pytorch",
        "rst_source": "docs/source/pytorch"
    }
]

# --- Process each repo config ---
for repo in REPOS:
    url = repo["url"]
    branch = repo["branch"]
    clone_dir = repo["clone_dir"]
    rst_source_dir = os.path.join(clone_dir, repo["rst_source"])

    # Clean previous clone
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)

    # Clone the repository
    print(f"Cloning {url} (branch: {branch})...")
    Repo.clone_from(url, clone_dir, branch=branch, depth=1)
    print("Cloned successfully.\n")

    # Extract .rst files
    print(f"Extracting .rst files from '{rst_source_dir}'...\n")
    for root, _, files in os.walk(rst_source_dir):
        for fname in files:
            if fname.endswith(".rst"):
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, rst_source_dir)
                flat_name = rel_path.replace(os.sep, "_")
                dst_path = os.path.join(".", f"{clone_dir}_{flat_name}")

                shutil.copyfile(src_path, dst_path)
                print(f" Saved: {dst_path}")

print("\nAll .rst files have been saved to the current directory.")
