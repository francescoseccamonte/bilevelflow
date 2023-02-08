#!/bin/sh
#
# Author : Francesco Seccamonte
# Copyright (c) 2023 Francesco Seccamonte. All rights reserved.
# Licensed under the Apache 2.0 License. See LICENSE file in the project root for full license information.
#

# Project setup, handling torch-geometric and its dependencies

TORCHV=1.11 # Fixing torch version for colab compatibility
CUDA=cpu     # Device to be used

# Allow user to pick torch version and cuda
while getopts h?:T:C: option; do
    case "${option}" in
    h|\?)
        echo "usage: $0 -T <torch version> -C <cuda version>"
        exit 0
        ;;
    T)
        TORCHV=${OPTARG}
        ;;
    C)  # If using cuda, it might be necessary adding --extra-index-url https://download.pytorch.org/whl/${CUDA} below
        CUDA=${OPTARG}
        ;;
    esac
done

pip install --upgrade pip
pip install torch==${TORCHV}
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCHV}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCHV}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCHV}+${CUDA}.html
pip install torch-geometric
pip install -e .
