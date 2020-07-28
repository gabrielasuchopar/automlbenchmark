#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh $HERE

PIP install --upgrade pip
PIP install --no-cache-dir genens==0.1.15

PIP install --no-cache-dir -r $HERE/requirements.txt
