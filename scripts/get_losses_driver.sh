#!/bin/bash
# create "outputs" folder if it doesn't exist
if [ ! -d "attack_data/outputs" ]; then
    mkdir "attack_data/outputs"
    echo "Created 'outputs' folder."
fi

# create "errors" folder if it doesn't exist
if [ ! -d "attack_data/errors" ]; then
    mkdir "attack_data/errors"
    echo "Created 'errors' folder."
fi

experiment_nos=( {0..32} )

epochs=(30)

datasets=("CIFAR10")

model_types=("vit_small_patch16_224")

# train non-DP models
for exp_no in "${experiment_nos[@]}"; do
    for data in "${datasets[@]}"; do
        for epoch_count in "${epochs[@]}"; do
            for model in "${model_types[@]}"; do
                clipping_mode="nonDP"
                eps=0.0
                sbatch get_losses.sh $exp_no $clipping_mode $eps $epoch_count $data $model
                sleep 0.75
            done
        done
    done
done