#!/bin/bash

# Convert dataset from v2.1 to v3.0
# Hardcoded to convert: thomas0829/eval_put_the_doll_into_the_box_2

REPO_ID=thomas0829/eval_car_box_depi

echo "Converting dataset: $REPO_ID from v2.1 to v3.0..."

python -m lerobot.common.datasets.v3.v30.convert_dataset_v21_to_v30 \
    --repo-id=$REPO_ID

echo "Conversion complete!"
