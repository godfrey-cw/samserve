#!/usr/bin/env bash

# # automatically install miniconda, dependencies
# mkdir ~/MiniCondaInstall && cd ~/MiniCondaInstall;
# curl -o install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh;
# bash install.sh -b -p $HOME/miniconda3;
# source ~/.bashrc;
# conda create --name=ts python=3.9;
# conda activate ts;
cd ~/samserve/serve;
python ts_scripts/install_dependencies.py --cuda=cu117; #need to figure out how to deal w/ sudo ...
pip install torchserve torch-model-archiver torch-workflow-archiver;
