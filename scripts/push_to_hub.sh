#!/bin/bash

# Push dataset to Hugging Face Hub
# Usage: ./push_to_hub.sh <repo_id>
# Example: ./push_to_hub.sh thomas0829/pick_up_the_toy_car_and_put_it_into_the_narrow_box

if [ -z "$1" ]; then
    echo "Usage: ./push_to_hub.sh <repo_id>"
    echo "Example: ./push_to_hub.sh thomas0829/my_dataset"
    exit 1
fi

REPO_ID="$1"

echo "Pushing dataset '$REPO_ID' to Hugging Face Hub..."

python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('$REPO_ID')
ds.push_to_hub()
print('Done! Dataset pushed to: https://huggingface.co/datasets/$REPO_ID')
"
