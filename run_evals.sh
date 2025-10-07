#!/bin/bash

# Check if a run_name argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <run_name>"
  exit 1
fi

run_name="$1"

# Define evaluation tasks and names
tasks=(
  "CPG-Rough-Unitree-A1-Eval-v0"
  "CPG-Rough-Unitree-A1-Eval-Discrete-v0"
  "CPG-Rough-Unitree-A1-Eval-Push-Flat-v0"
  "CPG-Rough-Unitree-A1-Eval-Push-Discrete-v0"
)

eval_names=(
  "ideal_flat"
  "ideal_discrete"
  "push_flat"
  "push_discrete"
)

# Ensure the number of tasks and eval_names match
if [[ "${#tasks[@]}" -ne "${#eval_names[@]}" ]]; then
  echo "Error: Number of tasks and eval_names do not match."
  exit 1
fi

# Loop through tasks and eval_names
for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  eval_name="${eval_names[$i]}"

  # Run the evaluation script
  python scripts/rsl_rl/eval.py \
    --task "$task" \
    --load_run "$run_name" \
    --eval_name "$eval_name" \
    --num_envs 1024 \
    --headless
done

echo "Evaluation completed."
