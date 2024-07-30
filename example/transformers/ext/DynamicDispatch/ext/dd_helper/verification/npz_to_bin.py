import numpy as np
import argparse, os


def dumpbins(file, dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # load data
    if os.path.exists(file) and file.endswith(".npz"):
        data = np.load(file)
        for name, tensor in data.items():
            binfile = name.replace("/", "_") + ".bin"
            with open(os.path.join(dirpath, binfile), "wb") as fp:
                fp.write(tensor.data)
    else:
        raise ValueError("- File not found. Path: {}".format(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input .npz file", type=str, required=True)
    parser.add_argument("--out-dir", help="Output dir path", type=str, required=True)
    args = parser.parse_args()

    dumpbins(args.file, args.out_dir)
