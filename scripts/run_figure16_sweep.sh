#!/bin/bash
# Figure 16 - MHA Parallelization Factor Sweep
# Part A: Sweep parfactor 1-64 on stream level 2
# Part B: Sweep stream level 1 with parfactor 1,2,4, then both levels with parfactor 4

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd $ARTIFACT_ROOT
export DATA_PATH=/tmp/data

PYTHON=python3
SCRIPT=$ARTIFACT_ROOT/scripts/run_end_to_end.py
MLIR=$ARTIFACT_ROOT/samml/tests/models/gpt-3/multihead_attention.mlir
OUTPUT_JSON=$ARTIFACT_ROOT/figure16_results.json

echo "=========================================="
echo "Figure 16a: MHA Parallelization Factor Sweep (Stream Level 2)"
echo "=========================================="
echo ""

# Initialize JSON file
echo '{' > $OUTPUT_JSON
echo '  "figure16a": {' >> $OUTPUT_JSON
echo '    "description": "Stream Level 2 parallelization sweep",' >> $OUTPUT_JSON
echo '    "stream_level": 2,' >> $OUTPUT_JSON
echo '    "results": {' >> $OUTPUT_JSON

FIRST=true

# Part A: Sweep parfactor 1-64 on stream level 2
for PAR in 1 2 4 8 16 32 64; do
    echo "=========================================="
    echo "Running MHA with streamlevel=2, parfactor=${PAR}"
    echo "=========================================="

    LOG=$ARTIFACT_ROOT/mha_sl2_par${PAR}.log

    $PYTHON $SCRIPT \
        --infile $MLIR \
        --build samml/build \
        -sp 0 \
        -par $PAR \
        -sl 2 \
        --block 64 \
        --useGen 2>&1 | tee $LOG

    # Extract cycles
    CYCLES=$(grep -i "Elapsed Cycles" $LOG 2>/dev/null | awk '{print $3}' || echo "0")

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo ',' >> $OUTPUT_JSON
    fi
    echo -n "      \"$PAR\": $CYCLES" >> $OUTPUT_JSON

    echo "streamlevel=2, parfactor=${PAR}: $CYCLES cycles"
    echo ""
done

echo '' >> $OUTPUT_JSON
echo '    }' >> $OUTPUT_JSON
echo '  },' >> $OUTPUT_JSON

echo ""
echo "=========================================="
echo "Figure 16b: MHA Stream Level 1 Sweep + Combined Levels"
echo "=========================================="
echo ""

echo '  "figure16b": {' >> $OUTPUT_JSON
echo '    "description": "Stream Level 1 parallelization sweep and combined levels",' >> $OUTPUT_JSON
echo '    "stream_level_1_results": {' >> $OUTPUT_JSON

FIRST=true

# Part B: Sweep stream level 1 with parfactor 1, 2, 4
for PAR in 1 2 4; do
    echo "=========================================="
    echo "Running MHA with streamlevel=1, parfactor=${PAR}"
    echo "=========================================="

    LOG=$ARTIFACT_ROOT/mha_sl1_par${PAR}.log

    $PYTHON $SCRIPT \
        --infile $MLIR \
        --build samml/build \
        -sp 0 \
        -par $PAR \
        -sl 1 \
        --block 64 \
        --useGen 2>&1 | tee $LOG

    # Extract cycles
    CYCLES=$(grep -i "Elapsed Cycles" $LOG 2>/dev/null | awk '{print $3}' || echo "0")

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo ',' >> $OUTPUT_JSON
    fi
    echo -n "      \"$PAR\": $CYCLES" >> $OUTPUT_JSON

    echo "streamlevel=1, parfactor=${PAR}: $CYCLES cycles"
    echo ""
done

echo '' >> $OUTPUT_JSON
echo '    },' >> $OUTPUT_JSON

# Part B continued: Parallelize both levels by a factor of 4
echo "=========================================="
echo "Running MHA with both levels parallelized (sl1=4, sl2=4)"
echo "=========================================="

LOG=$ARTIFACT_ROOT/mha_sl1_par4_sl2_par4.log

$PYTHON $SCRIPT \
    --infile $MLIR \
    --build samml/build \
    -sp 0 \
    -par 4 \
    -sl 1 \
    -par2 4 \
    -sl2 2 \
    --block 64 \
    --useGen 2>&1 | tee $LOG

CYCLES=$(grep -i "Elapsed Cycles" $LOG 2>/dev/null | awk '{print $3}' || echo "0")
echo "    \"combined_sl1_4_sl2_4\": $CYCLES" >> $OUTPUT_JSON

echo "streamlevel=1 par=4, streamlevel=2 par=4: $CYCLES cycles"
echo ""

echo '  }' >> $OUTPUT_JSON
echo '}' >> $OUTPUT_JSON

echo ""
echo "=========================================="
echo "Results saved to: $OUTPUT_JSON"
echo "=========================================="

echo ""
echo "=========================================="
echo "Summary of Results - Part A (Stream Level 2)"
echo "=========================================="
for PAR in 1 2 4 8 16 32 64; do
    LOG=$ARTIFACT_ROOT/mha_sl2_par${PAR}.log
    CYCLES=$(grep -i "Elapsed Cycles" $LOG 2>/dev/null | awk '{print $3}' || echo "N/A")
    echo "parfactor=${PAR}: ${CYCLES} cycles"
done

echo ""
echo "=========================================="
echo "Summary of Results - Part B (Stream Level 1)"
echo "=========================================="
for PAR in 1 2 4; do
    LOG=$ARTIFACT_ROOT/mha_sl1_par${PAR}.log
    CYCLES=$(grep -i "Elapsed Cycles" $LOG 2>/dev/null | awk '{print $3}' || echo "N/A")
    echo "parfactor=${PAR}: ${CYCLES} cycles"
done

echo ""
echo "Both levels (sl1=4, sl2=4):"
LOG=$ARTIFACT_ROOT/mha_sl1_par4_sl2_par4.log
CYCLES=$(grep -i "Elapsed Cycles" $LOG 2>/dev/null | awk '{print $3}' || echo "N/A")
echo "  ${CYCLES} cycles"
