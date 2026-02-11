#!/bin/bash

SCRIPT_DIR="./slurm_scripts"

for script in "$SCRIPT_DIR"/*.sh; do
    if [ -f "$script" ]; then
        echo "Submitting $script..."
        sbatch "$script"
    fi
done

