#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

DATASET_NAMES=(
  fb1k
  fb10k
  facebook
  nell-2
  nell-1
)

IGNORED_NAMES=(
  amazon-reviews
  reddit
  patents
)

FORMATS=(
  sss012
)

# Set these paths according to your environment
export FROSTT_PATH=${FROSTT_PATH:-/tmp/data/FROSTT}
export FROSTT_FORMATTED_PATH=${FROSTT_FORMATTED_PATH:-/tmp/data/FROSTT-formatted}
export FROSTT_FORMATTED_TACO_PATH=${FROSTT_FORMATTED_TACO_PATH:-/tmp/data/FROSTT-formatted/taco-tensor}
export TACO_TENSOR_PATH=${TACO_TENSOR_PATH:-/tmp/data}

basedir=$(pwd)

for i in ${!FORMATS[@]}; do
    format=${FORMATS[@]};
    echo "Generating files for format $format..."
    
    $basedir/compiler/taco/build/bin/taco-test sam.pack_$format
    $basedir/compiler/taco/build/bin/taco-test sam.pack_other_frostt

    for j in ${!DATASET_NAMES[@]}; do
        
        name=${DATASET_NAMES[$j]} 
        echo "Generating input format files for $name..."
        python $basedir/scripts/datastructure_tns.py -n $name -f $format
        python $basedir/scripts/datastructure_tns.py -n $name -f $format --other
        chmod -R 775 $FROSTT_FORMATTED_PATH
    done
done
