#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh $HERE

PIP install --upgrade pip
PIP install --no-cache-dir -U https://files.pythonhosted.org/packages/fb/43/d1d84ebc2ad591b15d91754c3213a32a9d81427f97763b3db239b03f54b2/genens-0.1.9-py3-none-any.whl

PIP install --no-cache-dir -r $HERE/requirements.txt
