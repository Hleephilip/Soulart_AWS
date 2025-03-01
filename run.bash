#!/bin/bash

cd alg_v5_diffusion_aws/
sudo -H -u ubuntu bash -c "python3 train_ours.py --config ./cfgs/dataset_cfgs/soulart.yaml"
