#!/bin/bash

cd movin_reproduction_aws/
sudo -H -u ubuntu bash -c "python3 train_movin.py --config ./cfgs/soulart.yaml"
