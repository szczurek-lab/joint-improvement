#bin/bash


python experiments/train/run.py \
    --output-dir results/pretrain/zinc250k/hyformer/pretrain_test \
    --train-dataset-config configs/datasets/zinc250k/unsupervised/train_dataset_config.json \
    --val-dataset-config configs/datasets/zinc250k/unsupervised/val_dataset_config.json \
    --tokenizer-config configs/tokenizers/smiles/zinc250k/tokenizer_config.json \
    --model-config configs/hyformer/base.json \
    --trainer-config configs/trainers/pretrain.json \
    --device cuda \
    --seed 0 \
    --num-workers 4 \
