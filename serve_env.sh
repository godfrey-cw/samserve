#!/usr/bin/env bash

sudo apt update && sudo apt upgrade;
# install torchserve + dependencies
cd ~/samserve/serve;
python ts_scripts/install_dependencies.py --cuda=cu117; # vm actually has 12.0 ...
pip install torchserve torch-model-archiver torch-workflow-archiver;

# install segment-anything
cd ~/samserve/segment-anything;
pip install -e .;

# make model archive file
cd ~;
mkdir models && mkdir samweights;
curl -o samweights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth;
torch-model-archiver --model-name sam-auto-maskgen --serialized-file samweights/sam_vit_b_01ec64.pth --model-file samserve/sam.py --handler samserve/sam_auto_maskgen.py  --export-path models/sam-auto-maskgen.mar -v 0.1;

# ssl stuff (note, requires interactive)
cd ~;
mkdir keys;
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout keys/mykey.key -out keys/mycert.pem;
curl -4 ifconfig.co >> keys/ipaddress.txt;

# launch torchserve
torchserve --model-store models --start --models all --ts-config samserve/config.properties --no-config-snapshots;