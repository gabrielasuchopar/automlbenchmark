#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh $HERE

SUDO apt-get update -y
SUDO apt-get install -y graphviz libgraphviz-dev

PIP install --upgrade pip
PIP install --no-cache-dir -U https://files.pythonhosted.org/packages/97/9c/e66fcda22a421761cd4792c4133757708d8352a28d8770652b10271e8829/genens-0.1.15-py3-none-any.whl

PIP install --no-cache-dir -r $HERE/requirements.txt
