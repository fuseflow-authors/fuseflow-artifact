import subprocess
from pathlib import Path
#import scorch
import re

PROJECT_ROOT_DIR = Path(__file__)
while not (PROJECT_ROOT_DIR / "samml-artifact").exists():
    PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent
ARTIFACT_ROOT = PROJECT_ROOT_DIR / "samml-artifact"
PROJECT_ROOT = PROJECT_ROOT_DIR / "samml-artifact" / "samml"


def run_command_list(command_list, cwd=None, debug=True):
    if debug:
        print("Running: ", " ".join(command_list))

    # Execute the command
    result = subprocess.run(command_list, cwd=cwd, capture_output=True, text=True)

    if debug:
        # Print the command's output
        print("STDOUT:", result.stdout)
        # Print the command's error output (if any)
        print("STDERR:", result.stderr)
        # Print the return code of the command
        print("Return Code:", result.returncode)
        print()
    return result


def run_command(command, cwd=None, debug=True):
    if debug:
        print("Running: ", command)

    # Execute the command
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, shell=True)

    if debug:
        # Print the command's output
        print("STDOUT:", result.stdout)
        # Print the command's error output (if any)
        print("STDERR:", result.stderr)
        # Print the return code of the command
        print("Return Code:", result.returncode)
        print()
    return result


# Assumes SAM MLIR files have comments that give the tensor name, mode, and index variables
def get_tensors_properties(samfile_path: Path):
    with samfile_path.open('r') as file:
        lines = file.readlines()

    assert "// BEGIN GENERATED MLIR CODE" in lines[0], ('SAM MLIR file must start with "// BEGIN GENERATED MLIR CODE": '
                                                        + str(lines[0]))
    lines = lines[1:]

    tensors = {}

    # Detect comment with tensor name
    i = 0
    while '//' in lines[i]:
        line = lines[i]
        # Comments take on the form: '// t0: dims(2) vars(i0, i1) mode_order(0, 1)'
        tensor_name, tensor_dict = parse_tensor_comment(line)
        if tensor_name is not None:
            tensors[tensor_name] = tensor_dict

        i += 1
    
    return tensors


import re


def parse_tensor_comment(line):
    # Regular expression to extract the elements
    pattern = r'// ((\w+)-([\d,]+)): dims\((\d+)\), vars\((.*?)\), mode_order\((.*?)\), format\((.*?)\), shape\((.*?)\)'
    pattern_with_block_arg = r'// ((\w+)-([\d,]+)): dims\((\d+)\), vars\((.*?)\), mode_order\((.*?)\), format\((.*?)\), shape\((.*?)\), BlockArg\((.*?)\)'
    match = re.match(pattern, line.strip())
    match_with_arg = re.match(pattern_with_block_arg, line.strip())

    if not match_with_arg and match:
        key = match.group(1)
        dims = int(match.group(4))
        vars_list = match.group(5).split(', ')
        mode_order = list(map(int, match.group(6).split(', ')))
        formats = match.group(7).split(', ')
        shape = match.group(8).split(', ')

        return key, {'dims': dims, 'vars': vars_list, 'mode_order': mode_order, 'format': formats, 'shape': shape, 'Arg': ''}
    
    if match_with_arg:
        key = match_with_arg.group(1)
        dims = int(match_with_arg.group(4))
        vars_list = match_with_arg.group(5).split(', ')
        mode_order = list(map(int, match_with_arg.group(6).split(', ')))
        formats = match_with_arg.group(7).split(', ')
        shape = match_with_arg.group(8).split(', ')
        blockArgNum = list(map(int, match_with_arg.group(9).split(', ')))[0]

        return key, {'dims': dims, 'vars': vars_list, 'mode_order': mode_order, 'format': formats, 'shape': shape, 'Arg': blockArgNum}

    return None, None

def check_gold(test_name: str, datadir_path: Path):
    if test_name == 'nested_matmuls':
        return True
    else:
        raise NotImplementedError
    return False
