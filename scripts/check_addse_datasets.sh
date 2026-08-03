#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPECTED_DIRS=(
    "data/external/EARS"
    "data/external/LibriSpeech"
    "data/external/VCTK"
    "data/external/DNS/datasets_fullband/clean_fullband"
    "data/external/MLS_URGENT_2025_track1"
    "data/external/WHAM_48kHz"
    "data/external/DEMAND"
    "data/external/FSD50K"
    "data/external/DNS/datasets_fullband/noise_fullband"
    "data/external/FMA_medium"
)

missing=0

cd "${ROOT_DIR}"

for dir in "${EXPECTED_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        echo "OK      ${dir}"
    else
        echo "MISSING ${dir}"
        missing=1
    fi
done

exit "${missing}"
