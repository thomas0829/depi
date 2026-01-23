accelerate launch --num_processes=1 lerobot/scripts/accelerate_train.py \
    --policy.path=/weka/oe-training-default/shiruic/depi/outputs/train/2026-01-22/07-50-44_rabc_finetuning/pretrained_model \
    --dataset.repo_id=thomas0829/put_the_doll_into_the_box \
    --batch_size=8 \
    --steps=1000 \
    --compile=true \
    --num_epochs=1 \
    --save_freq=1000 \
    --strict=true \
    --num_workers=4 \
    --log_freq=10 \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=0.0 \
    --optimizer.weight_decay=0.0 \
    --scheduler.type=diffuser \
    --scheduler.name=constant \
    --scheduler.num_warmup_steps=0 \
    --gradient_accumulation_steps=4 \
    --dataset.use_annotated_tasks=false \
    --dataset.force_cache_sync=true \
    --job_name=rabc_finetuning\
    --wandb.enable=false
    # sengi/DePi0 | Average loss: 0.0301
    # /weka/oe-training-default/shiruic/depi/outputs/train/2026-01-21/05-04-58_rabc_finetuning/pretrained_model | Average loss: 0.0283
    # sengi/depi_adv0 | Average loss: 0.0283
    # /weka/oe-training-default/shiruic/depi/outputs/train/2026-01-21/05-56-36_rabc_finetuning/checkpoints/last/pretrained_model | Average loss: 0.0284 #training adv0 dataset to convergence
    # sengi/depi_adv1 | Average loss: 0.0288 #training adv0 with adv0 rollout weighted by reward