#!/bin/bash

# Resolve the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# If we are running from workspace root, paths might be okay, but let's be robust.
# We will use absolute paths derived from SCRIPT_DIR.

INPUT_GENES="${SCRIPT_DIR}/data/iJO1366_genes.csv"
OUTPUT_DIR="${SCRIPT_DIR}/data"
CHUNK_DIR="${OUTPUT_DIR}/chunks"
NUM_WORKERS=8

# Ensure directories exist
mkdir -p "$CHUNK_DIR"

echo "Step 1: Splitting workload..."
python "${SCRIPT_DIR}/split_genes.py" \
    --input_file "$INPUT_GENES" \
    --num_chunks "$NUM_WORKERS" \
    --output_dir "$CHUNK_DIR"

echo "Step 2: Launching parallel workers..."
pids=""
for ((i=0; i<NUM_WORKERS; i++)); do
    echo "Starting worker $i..."
    # Running from SCRIPT_DIR ensures Python can find relative imports like 'cobra_models' if they are in that folder
    # However, 'cobra_models' is in 'experiments/01232026_fba/'. 
    # If we run python /abs/path/to/script.py, it adds that script's dir to sys.path.
    
    python "${SCRIPT_DIR}/predict_metabolic_ko_worker.py" \
        --input_csv "${CHUNK_DIR}/chunk_${i}.csv" \
        --output_dir "$OUTPUT_DIR" \
        --job_id "$i" > "${OUTPUT_DIR}/worker_${i}.log" 2>&1 &
    
    pids="$pids $!"
done

echo "Waiting for workers to finish (PIDs: $pids)..."
wait $pids

echo "Step 3: All workers finished. Proceeding to consolidation."
python "${SCRIPT_DIR}/consolidate_fba.py" --output_dir "$OUTPUT_DIR"

echo "Pipeline complete."
