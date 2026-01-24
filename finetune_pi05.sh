#!/bin/bash

n_gpus=1

export PRETRAINED_MODEL_PATH="thomas0829/pi05-pytorch-base"

RENAME_MAP='{"observation.images.front_camera":"observation.images.top","observation.images.left_camera":"observation.images.left","observation.images.right_camera":"observation.images.right"}'

gantry run --show-logs \
        --beaker-image shiruic/shirui-torch2.8.0_cuda12.8 \
        --cluster=ai2/ceres \
        --gpus=1 \
        --workspace=ai2/molmo-act \
        --weka=oe-training-default:/shiruic/depi \
        --name=training_depi_${TASK_NAME} \
        --priority=urgent \
        --install="uv pip install -e ".[pi05]"" \
        --env-secret HF_TOKEN=hf_token_shirui \
        --env-secret WANDB_API_KEY=wandb_token_shirui \
        -- \
        accelerate launch \
            --num_processes=${n_gpus} \
            lerobot/scripts/accelerate_train.py \
            --compile=true \
            --policy.path="${PRETRAINED_MODEL_PATH}" \
            --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}' \
            --rename_map="${RENAME_MAP}" \
            --dataset.repo_id="thomas0829/stack_red_cube_on_green_cube" \
            --dataset.image_transforms.enable=true \
            --batch_size=32 \
            --steps=20000 \
            --save_freq=5000 \
            --strict=true \
            --num_workers=4 \
            --log_freq=100 \
            --gradient_accumulation_steps=2 \
            --policy.gradient_checkpointing=true \
            --policy.scheduler_decay_lr=1e-5 \
            --policy.scheduler_decay_steps=1000000 \
            --policy.optimizer_lr=1e-4 \
            --policy.repo_id="thomas0829/finetune_pi05" \
            --policy.push_to_hub=true \
            --policy.private=false \
            --dataset.use_annotated_tasks=false \
            --job_name=pi05_training \
            --wandb.enable=true \
            --wandb.project="lerobot" \
            --wandb.entity="carl_uw" \
            --wandb.notes="Full fine-tuning of pi05 on put_the_dolls_on_the_cloth dataset"