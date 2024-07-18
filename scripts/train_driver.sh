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

# iterate through the settings
experiment_nos=( {0..32} )

epsilons=(0.5 1.0 2.0 8.0)

epochs=(30)

# other model_types: "vit_small_patch16_224" "vit_base_patch16_224" "vit_relpos_base_patch16_224.sw_in1k" "vit_relpos_small_patch16_224.sw_in1k"
# model_types=("vit_relpos_small_patch16_224.sw_in1k")
model_types=("vit_small_patch16_224")

# train non-DP models
for exp_no in "${experiment_nos[@]}"; do
    for epoch_count in "${epochs[@]}"; do
        for model_type in "${model_types[@]}"; do
            clipping_mode="nonDP"
            sbatch train.sh $exp_no $clipping_mode $epoch_count $model_type
            sleep 0.5
            fi
        done
    done
done