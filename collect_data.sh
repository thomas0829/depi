#!/bin/bash

# Default repo_id
REPO_ID="thomas0829/eval_car_box_depi_v2"
CACHE_DIR="/home/prior/.cache/huggingface/lerobot/${REPO_ID}"

# Check if dataset already exists
if [ -d "$CACHE_DIR" ]; then
    while true; do
        echo "Dataset already exists at: $CACHE_DIR"
        echo "What would you like to do?"
        echo "  [d] Delete existing dataset and continue"
        echo "  [r] Rename and use a new repo_id"
        echo "  [q] Quit"
        read -p "Your choice: " choice
        
        case $choice in
            d|D)
                echo "Deleting existing dataset..."
                rm -rf "$CACHE_DIR"
                echo "Deleted. Continuing with recording..."
                break
                ;;
            r|R)
                while true; do
                    read -p "Enter new dataset name (e.g., new_name): " NEW_NAME
                    if [ -z "$NEW_NAME" ]; then
                        echo "No name provided. Please try again."
                    else
                        REPO_ID="thomas0829/$NEW_NAME"
                        echo "Using new repo_id: $REPO_ID"
                        break
                    fi
                done
                break
                ;;
            q|Q)
                echo "Exiting."
                exit 0
                ;;
            *)
                echo "Invalid choice. Please enter d, r, or q."
                ;;
        esac
    done
fi

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the toy car and put it into the narrow box." \
  --control.num_episodes=6 \
  --control.warmup_time_s=5 \
  --control.episode_time_s=90 \
  --control.reset_time_s=5 \
  --control.push_to_hub=true \
  --control.repo_id="$REPO_ID" \
  --control.policy.device=cuda \
  --control.policy.path=sengi/DePi0
  