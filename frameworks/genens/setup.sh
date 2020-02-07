#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
PIP install --no-cache-dir -U https://files.pythonhosted.org/packages/16/ae/4bab52c593cad635bcb3e09a9586a047a515fd053188ed7915bf0b5b1482/genens-0.1.8-py3-none-any.whl
PIP install --no-cache-dir -r $HERE/requirements.txt
