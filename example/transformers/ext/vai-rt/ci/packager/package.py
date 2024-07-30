import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source", help="Path of the source directory")
parser.add_argument("destination", help="Path of the destination directory")
parser.add_argument("file_list", help="Path of the file containing the list of files to copy")
args = parser.parse_args()

if not os.path.exists(args.destination):
    os.makedirs(args.destination)

with open(args.file_list) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('-'):
            # Remove files or folders
            line = line[1:].strip()
            path = os.path.join(args.destination, line)
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            # Copy files or folders
            path = os.path.join(args.source, line)
            if '*' in line:
                # Include all files and folders in a directory
                path = os.path.join(args.source, os.path.dirname(line))
                if os.path.exists(path):
                    shutil.copytree(path, os.path.join(args.destination, os.path.dirname(line)))
            else:
                # Include a single file or folder
                if os.path.exists(path):
                    dest_path = os.path.join(args.destination, line)
                    if os.path.isdir(path):
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)
                        shutil.copytree(path, dest_path)
                    else:
                        if not os.path.exists(os.path.dirname(dest_path)):
                            os.makedirs(os.path.dirname(dest_path))
                        shutil.copy(path, dest_path)

