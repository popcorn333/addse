#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/external"

mkdir -p "${DATA_DIR}"
cd "${ROOT_DIR}"

missing=0

link_dataset() {
    local name="$1"
    local source="$2"
    local dest="$3"

    if [[ ! -d "${source}" ]]; then
        echo "MISSING ${name}: source not found: ${source}"
        echo "MISSING ${name}: expected destination: ${dest}"
        missing=1
        return 0
    fi

    if [[ -e "${dest}" && ! -L "${dest}" ]]; then
        echo "SKIP    ${name}: ${dest} already exists and is not a symlink"
        return 0
    fi

    mkdir -p "$(dirname "${dest}")"
    ln -sfn "${source}" "${dest}"
    echo "LINKED  ${name}: ${dest} -> ${source}"
}

missing_dataset() {
    local name="$1"
    local readme_link="$2"
    local dest="$3"

    if [[ -d "${dest}" ]]; then
        echo "OK      ${name}: ${dest} already exists"
        return 0
    fi

    echo "MISSING ${name}"
    echo "        README/source link: ${readme_link}"
    echo "        expected destination: ${dest}"
    missing=1
}

missing_dataset \
    "EARS" \
    "https://github.com/facebookresearch/ears_dataset" \
    "${DATA_DIR}/EARS"

missing_dataset \
    "LibriSpeech" \
    "https://www.openslr.org/12" \
    "${DATA_DIR}/LibriSpeech"

link_dataset \
    "VCTK" \
    "/mnt/parscratch/users/acp23xt/public/VCTK_22k" \
    "${DATA_DIR}/VCTK"

missing_dataset \
    "DNS5 clean speech" \
    "https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh" \
    "${DATA_DIR}/DNS/datasets_fullband/clean_fullband"

missing_dataset \
    "MLS_URGENT_2025_track1" \
    "https://huggingface.co/datasets/kohei0209/mls_hq_urgent_track1" \
    "${DATA_DIR}/MLS_URGENT_2025_track1"

missing_dataset \
    "WHAM_48kHz" \
    "http://wham.whisper.ai/" \
    "${DATA_DIR}/WHAM_48kHz"

missing_dataset \
    "DEMAND" \
    "https://zenodo.org/records/1227121" \
    "${DATA_DIR}/DEMAND"

missing_dataset \
    "FSD50K" \
    "https://zenodo.org/records/4060432" \
    "${DATA_DIR}/FSD50K"

missing_dataset \
    "DNS noise" \
    "https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh" \
    "${DATA_DIR}/DNS/datasets_fullband/noise_fullband"

missing_dataset \
    "FMA_medium" \
    "https://github.com/mdeff/fma" \
    "${DATA_DIR}/FMA_medium"

exit "${missing}"
