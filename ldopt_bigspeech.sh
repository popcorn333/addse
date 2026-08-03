#!/bin/bash
uv run addse ldopt \
    data/external/EARS/ \
    data/external/VCTK/ \
    --regexes "^.*/p0[0-9][0-9]/.*\.wav$" \
    --regexes "^.*\.flac$" \
    --labels ears \
    --labels vctk \
    data/chunks/bigspeech/ \
    --num-workers 4 \
    --seed 42
