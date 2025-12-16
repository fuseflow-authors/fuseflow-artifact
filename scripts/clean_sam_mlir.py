from pathlib import Path
import argparse


def process_sam_mlir_file(filepath: Path, print_out: bool = True):
    # Read the content of the file
    with filepath.open('r') as file:
        lines = file.readlines()

    # Find the line containing "// BEGIN"
    begin_index = None
    for i, line in enumerate(lines):
        if "// BEGIN GENERATED MLIR CODE" in line:
            begin_index = i
            break

    # If "// BEGIN" is found, write the remaining lines back to the file
    if begin_index is not None:
        with filepath.open('w') as file:
            file.writelines(lines[begin_index:])

    if print_out:
        print('Processing Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default=None,
                        help='The file to remove lines from (usually a SAM(ML) MLIR file)')
    parser.add_argument('--print',  '-p', action='store_true',
                        help='Print in progress information')
    args = parser.parse_args()

    # Example usage
    file_path = Path(args.infile)

    if args.print:
        print("Processing file:", file_path)
    process_sam_mlir_file(file_path, print_out=args.print)
