#!/usr/bin/env bash

VENVNAME=vis_analytics_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt

echo "finished building $VENVNAME"
