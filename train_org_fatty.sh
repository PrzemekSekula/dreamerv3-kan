#!/bin/bash

############################################
# Default parameters
############################################
LOG_FOLDER_DEFAULT="kan_org"
# Define a default list of GPU IDs to use
GPU_LIST_DEFAULT=(1 2 3 5 6)
RUNS_PER_GPU_DEFAULT=3

############################################
# Parse command-line arguments
############################################
LOG_FOLDER="$LOG_FOLDER_DEFAULT"
# Copy the default array
GPU_LIST=("${GPU_LIST_DEFAULT[@]}")
RUNS_PER_GPU="$RUNS_PER_GPU_DEFAULT"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --logfolder)
      LOG_FOLDER="$2"
      shift 2
      ;;
    --runs_per_gpu)
      RUNS_PER_GPU="$2"
      shift 2
      ;;
    --gpulist)
      # Convert comma-separated string (e.g., "1,2,5") to a bash array
      GPU_LIST=($(echo "$2" | tr ',' ' '))
      shift 2
      ;;
    *)
      echo "Unrecognized parameter: $1"
      echo "Usage: $0 [--logfolder FOLDER] [--runs_per_gpu M] [--gpulist L]"
      exit 1
      ;;
  esac
done

echo "Using LOG_FOLDER='${LOG_FOLDER}'"
echo "Using RUNS_PER_GPU='${RUNS_PER_GPU}'"
echo "Using GPU_LIST='${GPU_LIST[@]}'"

############################################
# List of tasks (Atari environments)
############################################
TASKS=(
  ms_pacman
  qbert
  alien
  kangaroo
  asterix
  pong
)

############################################
# We want runs_per_gpu tasks per GPU
############################################
# Get the number of GPUs from the length of the GPU_LIST array
GPU_COUNT=${#GPU_LIST[@]}
SLOTS_COUNT=$((GPU_COUNT * RUNS_PER_GPU))

echo "Total number of slots (SLOTS_COUNT): $SLOTS_COUNT"


# Initialize an array of length SLOTS_COUNT
# Each element holds the PID of the process occupying that slot.
# If the value is 0, that slot is free.
declare -a SLOT_PIDS
for ((slot=0; slot<SLOTS_COUNT; slot++)); do
  SLOT_PIDS[$slot]=0
done

############################################
# Function to start a single training run
############################################
start_training() {
  local task=$1
  local slot_id=$2

  # Calculate index in GPU_LIST array: floor(slot_id / RUNS_PER_GPU)
  local gpu_index=$((slot_id / RUNS_PER_GPU))
  # Get the actual GPU ID from the array
  local gpu_id=${GPU_LIST[$gpu_index]}

  echo "Starting atari_${task} on cuda:${gpu_id} (slot ${slot_id})"
  python3 dreamer.py \
    --configs atari100k \
    --task atari_"${task}" \
    --logdir ./log_atari100k/"${LOG_FOLDER}"/"${task}" \
    --device cuda:"${gpu_id}" &

  # Store the PID of the newly launched process
  SLOT_PIDS[$slot_id]=$!
}

############################################
# Main loop over tasks
############################################
for task in "${TASKS[@]}"; do
  while true; do
    # Check each slot in [0..SLOTS_COUNT-1]
    for ((slot=0; slot<SLOTS_COUNT; slot++)); do
      # If SLOT_PIDS[slot] == 0, that slot is free
      if [ "${SLOT_PIDS[$slot]}" -eq 0 ]; then
        start_training "$task" "$slot"
        # Break out of the slot loop, go to next task
        break 2
      fi

      # If the slot is not free, check if the process is still alive
      if ! kill -0 "${SLOT_PIDS[$slot]}" 2>/dev/null; then
        # Process ended, so the slot is now free
        start_training "$task" "$slot"
        break 2
      fi
    done

    # If we did NOT break, it means no slot is free => wait a bit, then check again
    sleep 2
  done
done

############################################
# Wait for all remaining training to finish
############################################
wait
echo "All training runs have completed!"