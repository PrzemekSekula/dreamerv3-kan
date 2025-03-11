#!/usr/bin/env bash

############################################
# Default parameters
############################################
LOG_FOLDER_DEFAULT="original/seed_1"
GPU_COUNT_DEFAULT=9

############################################
# Parse command-line arguments
############################################
LOG_FOLDER="$LOG_FOLDER_DEFAULT"
GPU_COUNT="$GPU_COUNT_DEFAULT"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --logfolder)
      LOG_FOLDER="$2"
      shift 2
      ;;
    --gpucount)
      GPU_COUNT="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized parameter: $1"
      echo "Usage: $0 [--logfolder FOLDER] [--gpucount N]"
      exit 1
      ;;
  esac
done

echo "Using LOG_FOLDER='${LOG_FOLDER}'"
echo "Using GPU_COUNT='${GPU_COUNT}'"

############################################
# List of tasks (Atari environments)
############################################
TASKS=(
  up_n_down
  crazy_climber
  battle_zone
  breakout
  private_eye
  bank_heist
  kung_fu_master
  freeway
  pong
  hero
  boxing
  gopher
  krull
  chopper_command
  demon_attack
  seaquest
  road_runner
  assault
  frostbite
  amidar
  jamesbond
  ms_pacman
  qbert
  alien
  kangaroo
  asterix
)

############################################
# We want 2 tasks per GPU in parallel
# => total slots = 2 * GPU_COUNT
############################################
SLOTS_COUNT=$((GPU_COUNT * 2))

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

  # The actual GPU to use is floor(slot_id / 2)
  # because each GPU has 2 slots: (0,1) -> gpu 0, (2,3) -> gpu 1, etc.
  local gpu_id=$((slot_id / 2))

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

      # Else, check if the process in that slot is still alive
      if ! kill -0 "${SLOT_PIDS[$slot]}" 2>/dev/null; then
        # The process ended, so the slot is now free
        start_training "$task" "$slot"
        break 2
      fi
    done

    # If we did NOT break, it means no slot was free => wait a bit and check again
    sleep 2
  done
done

############################################
# Wait for all remaining training to finish
############################################
wait
echo "All training runs have completed!"
