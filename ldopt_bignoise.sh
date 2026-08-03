#!/bin/bash
uv run addse ldopt \
    data/external/WHAM_48kHz/ \
    data/external/DEMAND/ \
    --regexes "^.*/file(0[0-9][0-9]|1[0-7][0-9]|18[0-8]).*\.wav$" \
    --regexes "^.*\.wav$" \
    --labels wham \
    --labels demand \
    --seglens 30.0 \
    --seglens 10.0 \
    data/chunks/bignoise/ \
    --num-workers 4 \
    --seed 42
