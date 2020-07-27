#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh $HERE

SUDO apt-get update -y
SUDO apt-get install -y graphviz libgraphviz-dev

PIP install --upgrade pip
PIP install --no-cache-dir genens==0.1.15

PIP install --no-cache-dir -r $HERE/requirements.txt
