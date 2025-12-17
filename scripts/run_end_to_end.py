import subprocess
import os
import argparse
import data_generator
import sys

import comal

from pathlib import Path
from utils import *
from clean_sam_mlir import process_sam_mlir_file
import py

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matfile', type=str, default=None,
                        help='The folder of the matrices (assumed to be in .mtx format). If it is None, will use '
                             'synthetically generated data')
    parser.add_argument('-mdir', '--matdir', type=str, default=None,
                        help='The path of the directory that contains all of the formatted tensor data')
    parser.add_argument('-par', '--parfactor', type=int, default=1,
                        help='The parallel factor')
    parser.add_argument('-sl', '--streamlevel', type=int, default=1,
                        help='The stream level for parallelization (dimension to parallelize)')
    parser.add_argument('-par2', '--parfactor2', type=int, default=None,
                        help='Second parallel factor (for multi-level parallelization)')
    parser.add_argument('-sl2', '--streamlevel2', type=int, default=None,
                        help='Second stream level (for multi-level parallelization)')
    parser.add_argument('-s', '--shape', type=int, default=10,
                        help='The shape of the tensor')
    parser.add_argument('-sp', '--sparsity', type=float, default=0.80,
                        help='The sparsity of the generated tensor')
    parser.add_argument('--skewness', type=float, default=0.0,
                        help='Power-law exponent for degree distribution (0=uniform, 2-3=scale-free)')
    parser.add_argument('--seed', type=int, default=25,
                        help='The seeds used to generate tensors')
    parser.add_argument('--block', type=int, default=64,
                        help='Block size for BigBird')
    parser.add_argument('--outformat', '-oformat', type=str, default='CSF',
                        help='The output format')
    parser.add_argument('--infile', type=str,
                        help='The filename of the SparseTensor MLIR file to compile and simulate from PROJECT_DIR',
                        required=True)
    parser.add_argument('-b', '--build', type=str, default=None, help='Path to build directory. '
                                                                'If not set, will use build/ in ARTIFACT_HOME/samml')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Print out more information if in debug mode')
    parser.add_argument('--inDataset', '-inDataset', type=str, default="",
                        help='DatasetName set')
    parser.add_argument('--inData', '-inData', type=str, default="",
                        help='data name')
    parser.add_argument('--useGen', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Use generator')
    parser.add_argument('--checkGold', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Run gold checking function on tmp result files')
    parser.add_argument('--blockmode', type=int, default=1, choices=[1, 16, 32, 64],
                        help='Block size for comal simulation (1=scalar, 16/32/64=block sparse)')
    parser.add_argument('--trueblock', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Use true block sparse simulation (BCSR format with NxN dense blocks)')
    parser.add_argument('--loop-order-index', type=int, default=-1,
                        help='Non-interactive selection of dataflow loop order index (0-based). Use -1 for default.')
    parser.add_argument('--skip-iterate-locate', action='store_true', default=False,
                        help='Skip the InsertIterateLocate pass (for benchmarks that don\'t need dense tensor optimizations)')
    return parser


def run_samml_all(args):
    result = dict()
    # Generate synthetic tensors
    
    if args.checkGold:
        match = data_generator.check_gold("/tmp/tensor_mha_out_mode_vals", "/tmp/tmp_mha_result.txt")
        print("Matches: ", match)
        exit(0)

    # [sam-opt] Run SparseTensor to SAMML in MLIR
    # [sam-translate] Emit proto in MLIR

    # Create a Path object
    inpath = Path(args.infile)
    in_abspath = inpath.resolve()

    #assert in_abspath.is_relative_to(PROJECT_ROOT), ("Test files (" + str(in_abspath) +
    #                                                 ") must be in project path: " + PROJECT_ROOT)
    print("Input filename:", in_abspath)
    test_name = str(in_abspath).split("/")[-1].replace(".mlir", "")
    if len(args.inDataset) > 0 or len(args.inData) > 0:
        data_class = getattr(data_generator, test_name)(args)
        use_data_set = True
    elif args.useGen:
        # Try blocked version first (for true block sparse), then synthetic, then regular
        blocked_name = test_name + "_blocked"
        synthetic_name = test_name + "_synthetic"
        if args.trueblock and hasattr(data_generator, blocked_name):
            print(f"Using blocked data generator: {blocked_name}")
            data_class = getattr(data_generator, blocked_name)(args)
        elif hasattr(data_generator, synthetic_name):
            data_class = getattr(data_generator, synthetic_name)(args)
        else:
            data_class = getattr(data_generator, test_name)(args)
        use_data_set = True
    else:
        use_data_set = False

    print("Test name", test_name)
    test_type = in_abspath.parent.name
    test_name = in_abspath.stem

    # Use separate data directory for true block sparse mode AND per block size
    # This ensures scalar mode data with block_size=16 doesn't get overwritten by block_size=32/64
    if args.trueblock:
        data_test_name = f"{test_name}_blocked_b{args.block}"
    else:
        data_test_name = f"{test_name}_b{args.block}"

    # Create the new filename by appending '_sam' and re-adding the original extension
    samfile_abspath = in_abspath.parent/f"{test_name}_sam{in_abspath.suffix}"
    
    # samfile = f"{in_abspath.parent}/{in_abspath.stem}_sam{in_abspath.suffix}"
    print("SAMML filename:", samfile_abspath)

    # Run sam-opt either from binary or from default location
    command = './tools/sam-opt'
    working_dir = PROJECT_ROOT/"build" if args.build is None or args.build == 'None' else args.build
    if args.debug:
        print("sam-opt working dir:", working_dir)

    # Build parallelizer pass string (supports multi-level parallelization)
    parallelizer_passes = f'--stream-parallelizer="stream-level={args.streamlevel} par-factor={args.parfactor}"'
    if args.parfactor2 is not None and args.streamlevel2 is not None:
        parallelizer_passes += f' --stream-parallelizer="stream-level={args.streamlevel2} par-factor={args.parfactor2}"'

    # Build linalg-to-sam pass with optional loop order index and skip-iterate-locate
    linalg_to_sam_opts = []
    if args.loop_order_index >= 0:
        linalg_to_sam_opts.append(f'loop-order-index={args.loop_order_index}')
    if args.skip_iterate_locate:
        linalg_to_sam_opts.append('skip-iterate-locate=true')

    if linalg_to_sam_opts:
        linalg_to_sam_pass = f'--linalg-to-sam="{" ".join(linalg_to_sam_opts)}"'
    else:
        linalg_to_sam_pass = '--linalg-to-sam'

    opt_out = run_command(command + f' {linalg_to_sam_pass} -cse {parallelizer_passes} ' +
            str(in_abspath) + ' > ' + str(samfile_abspath),
            cwd=working_dir,
            debug=True)
    result["sam-opt"] = opt_out
    process_sam_mlir_file(samfile_abspath, print_out=True)

    # Create proto generation location
    protodir_path = ARTIFACT_ROOT/"proto"/test_type
    # Create the directory if it does not exist
    protodir_path.mkdir(parents=True, exist_ok=True)
    protofile_path = protodir_path/f"{test_name}_proto.proto"
    print("Proto filename:", protofile_path)

    command = './tools/sam-translate'
    working_dir = PROJECT_ROOT/"build" if args.build is None or args.build == 'None' else args.build
    if args.debug:
        print("sam-translate working dir:", working_dir)

    translate_out = run_command(command + ' ' + str(samfile_abspath) + ' --emit-proto > ' + str(protofile_path),
                cwd=working_dir,
                debug=True)
    result["sam-translate"] = translate_out

    # Immediately copy op.bin to protodir to avoid race conditions with parallel runs
    import shutil
    tmp_op_path = "/tmp/op.bin"
    op_bin_path = protodir_path / f"{test_name}_op.bin"
    if Path(tmp_op_path).exists():
        shutil.copy(tmp_op_path, op_bin_path)
        print(f"Copied op.bin to: {op_bin_path}")

    samgraph_path = protodir_path/f"{test_name}_graph.png"
    working_dir = ARTIFACT_ROOT/"tortilla-visualizer"
    if args.debug:
        print("get_dot.py working dir:", working_dir)
    print("SAM Graph PNG:", samgraph_path)

    dot_out = run_command(f'{sys.executable} get_dot.py -f ' + str(protofile_path) + ' -o ' + str(samgraph_path),
                cwd=working_dir,
                debug=True)
    result["sam-graph"] = dot_out

    matdir = args.matdir
    print("MATDIR IF FOUND IS:  ", matdir)
    # Generate data if not already generated
   
    if matdir is None:
        # Create the matrix data directory (use data_test_name to separate scalar vs blocked)
        matdir = ARTIFACT_ROOT/"data"/test_type/data_test_name
        matdir.mkdir(parents=True, exist_ok=True)
        print("Data Directory (MATDIR):", matdir)

        shape = args.shape
        seed = args.seed
        sparsity = args.sparsity
        output_format = args.outformat
        shape_map = {}
        tensors = get_tensors_properties(samfile_abspath)
        
        # Running data generation command for each tensor
        for tensor_name, tensor_property in tensors.items():
            if use_data_set:
                print(f"tensor name, shape and mode, {tensor_name}, ", tensor_property["shape"], tensor_property["mode_order"])
                data_class.collect_names(tensor_name, tensor_property["Arg"], tensor_property["mode_order"], tensor_property["shape"], tensor_property["format"])
            else:

                print(f"tensor name and shape, {tensor_name}, shape:", end="")
                for shape in tensor_property.get("shape"):
                    print(str(shape), end=",")
                print("")

                gendata_filename = str(ARTIFACT_ROOT/"sam"/"sam"/"onyx"/"synthetic"/"generate_random_mats.py")
                command_list = [sys.executable, gendata_filename, '--output_dir', str(matdir), '--name', tensor_name, '--shape']
                for shape in tensor_property.get('shape'):
                    command_list.append(str(shape))
                command_list.append('--mode_ordering')
                for mode in tensor_property.get('mode_order'):
                    command_list.append(str(mode))
                command_list.append('--sparsity')
                command_list.append(str(sparsity))
                command_list.append('--seed')
                command_list.append(str(seed))
                command_list.append('--output_format')
                command_list.append(output_format)
                tensor_out = run_command_list(command_list, debug=True)
                result["tensor-"+tensor_name] = tensor_out
    if use_data_set:
        print("GENERATE DATA")
        # Set the file_path to match matdir
        data_class.file_path = "/" + str(matdir.relative_to(ARTIFACT_ROOT)) + "/"
        # Pass sparsity to generate_data if the method supports it
        try:
            data_class.generate_data(weight_sparsity=args.sparsity)
        except TypeError:
            # Fallback for data classes that don't support weight_sparsity
            data_class.generate_data()
    print("Running Comal")
    # [Comal]
    # Use the op.bin saved in protodir (to avoid race conditions with parallel runs)
    local_op_path = str(matdir / "op.bin")
    if op_bin_path.exists():
        shutil.copy(op_bin_path, local_op_path)
    else:
        # Fallback to /tmp/op.bin if protodir op.bin doesn't exist
        tmp_op_path = "/tmp/op.bin"
        shutil.copy(tmp_op_path, local_op_path)
    # should be a tuple of (passed: bool, cycles:int)
    # Select the appropriate comal function based on blockmode and trueblock
    try:
        if args.trueblock:
            # True block sparse: uses BCSR format with NxN dense Tensor blocks
            if args.blockmode == 16:
                sim_output = comal.run_graph_true_block16(local_op_path, str(matdir))
            elif args.blockmode == 32:
                sim_output = comal.run_graph_true_block32(local_op_path, str(matdir))
            elif args.blockmode == 64:
                sim_output = comal.run_graph_true_block64(local_op_path, str(matdir))
            else:
                print(f"True block sparse requires blockmode 16/32/64, got {args.blockmode}")
                sim_output = (False, None)
        else:
            # Timing-only block sparse: scalar values with block timing
            if args.blockmode == 1:
                sim_output = comal.run_graph(local_op_path, str(matdir))
            elif args.blockmode == 16:
                sim_output = comal.run_graph_block16(local_op_path, str(matdir))
            elif args.blockmode == 32:
                sim_output = comal.run_graph_block32(local_op_path, str(matdir))
            elif args.blockmode == 64:
                sim_output = comal.run_graph_block64(local_op_path, str(matdir))
            else:
                print(f"Unknown blockmode: {args.blockmode}, using scalar mode")
                sim_output = comal.run_graph(local_op_path, str(matdir))
    except comal.PanicError:
        sim_output = (False, None)

    print("Simulation completed", sim_output)
    result["comal"] = sim_output

    # TODO: gold check against scorch
    #check_gold(test_name, matdir)

    if args.debug:
        print("Output dictionary:", result)

    return result

if __name__ == "__main__":
    print(sys.argv)
    parse = get_argparser()
    args = parse.parse_args()
    run_samml_all(args)
