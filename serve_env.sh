#!/usr/bin/env bash

# install torchserve + dependencies
cd ~/samserve/serve;
python ts_scripts/install_dependencies.py --cuda=cu117; # vm actually has 12.0 ...
pip install torchserve torch-model-archiver torch-workflow-archiver;

# install segment-anything
cd ~/samserve/segment-anything;


# launch torchserve
cd ~;
mkdir models && mkdir workflows;
torchserve --start --model-store=models --workflow-store=workflows;