import os 
import subprocess
from utils import *
import argparse

parallel = True



# Git submodule update
git_command = ['git', 'submodule', 'update', '--init', '--recursive']
run_command_list(git_command, cwd=ARTIFACT_ROOT)

# Run external llvm build
llvm_cwd = ARTIFACT_ROOT/'fuseflow-compiler/external/llvm-project/'

llvm_cmake = 'cmake -G Ninja ../llvm' + \
   '-DLLVM_ENABLE_PROJECTS=mlir ' + \
   '-DLLVM_TARGETS_TO_BUILD="Native;" ' + \
   '-DCMAKE_BUILD_TYPE=RelWithDebInfo ' + \
   '-DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ' + \
   '-DLLVM_ENABLE_LLD=ON && cmake --build . --target tools/mlir/test/check-mlir'
if parallel:
    llvm_cmake += ' --parallel 8'

llvm_command = ['mkdir', 'build', '&&', 'cd', 'build', '&&'] + llvm_cmake.split(' ')
run_command_list(llvm_command, cwd=llvm_cwd)

# Run or-tools build
# Linux
# sudo apt update
# sudo apt install -y build-essential cmake lsb-release
