import numpy as np
import os
import argparse


def write_raw(npy_file, raw_file):
    # Load data from npy file
    array = np.load(npy_file)
    # Save data as raw file
    with open(raw_file, "wb") as f:
        array.tofile(f)


def npy_to_raw(input_dir, output_dir, prefix=""):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            npy_file = os.path.join(input_dir, file_name)
            raw_file = os.path.join(
                output_dir, prefix + os.path.splitext(file_name)[0] + ".raw"
            )
            write_raw(npy_file, raw_file)


def main():
    parser = argparse.ArgumentParser(description="Convert .npy files to .raw files")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing PSJ input data directories",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory to save .raw files"
    )
    parser.add_argument(
        "--single",
        default=False,
        action="store_true",
        help="For Models with single input",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each npy file in the input directory
    try:
        if args.single:
            npy_to_raw(input_dir, output_dir)
        else:
            npy_to_raw(os.path.join(input_dir, "embedding"), output_dir, "embeddings_")
            npy_to_raw(
                os.path.join(input_dir, "float_attention_mask"),
                output_dir,
                "attention_mask_",
            )
        print("\n- Data files stored in : {}\n".format(output_dir))
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
