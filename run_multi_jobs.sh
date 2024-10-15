#!/bin/bash

# Array of MAX_TURNS values to iterate over
max_turns_values=(1 2 3 5)

# Define chunk size
k=1200

# Calculate start indices based on chunk size
start_indices=(0 $k $((2 * k)) $((3 * k)))

# Loop over the max_turns_values and start_indices arrays to submit jobs
for MAX_TURNS in "${max_turns_values[@]}"
do
  for i in "${!start_indices[@]}"
  do
    START_IDX=${start_indices[$i]}
    END_IDX=$((START_IDX + k))

    # Export variables as environment variables for the job
    export MAX_TURNS=$MAX_TURNS
    export START_IDX=$START_IDX
    export END_IDX=$END_IDX

    # Submit the SLURM job
    sbatch --mem=65536M main.sh --export=ALL

    # Optional: add a short delay to avoid overwhelming the scheduler
    sleep 1
  done
done
