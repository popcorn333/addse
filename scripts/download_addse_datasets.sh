#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/parscratch/users/acp23xt/private/codex_addse/addse"
DATA_DIR="${ROOT_DIR}/data/external"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/download_addse_datasets.log"

mkdir -p "${DATA_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
cd "${ROOT_DIR}"

echo "==== ADDSE four-dataset download started: $(date -Is) ===="
echo "Data: ${DATA_DIR}"
echo "Log: ${LOG_FILE}"

is_populated_dir() {
    [[ -d "$1" ]] && find "$1" -mindepth 1 -print -quit | grep -q .
}

download_file() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "${dest}")"
    if [[ -s "${dest}" ]]; then
        echo "SKIP existing file: ${dest}"
        return 0
    fi
    echo "DOWNLOAD ${url}"
    echo "       -> ${dest}"
    if command -v wget >/dev/null 2>&1; then
        wget -c --tries=10 --timeout=60 -O "${dest}" "${url}"
    else
        curl -L -C - --retry 10 --retry-delay 10 -o "${dest}" "${url}"
    fi
}

extract_zip_once() {
    local archive="$1"
    local dest="$2"
    local marker="${dest}/.extracted.$(basename "${archive}")"
    if [[ -f "${marker}" ]] && is_populated_dir "${dest}"; then
        echo "SKIP extracted: ${dest}"
        return 0
    fi
    mkdir -p "${dest}"
    unzip -q -o "${archive}" -d "${dest}"
    touch "${marker}"
}

download_vctk() {
    local archive="${DATA_DIR}/VCTK-Corpus-0.92.zip"
    local target="${DATA_DIR}/VCTK"

    if [[ -L "${target}" ]]; then
        echo "ERROR VCTK target is a symlink: ${target}"
        exit 1
    fi

        if is_populated_dir "${target}"; then
        echo "SKIP VCTK: ${target} already populated"
        return 0
    fi

    download_file "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip" "${archive}"

    local tmp="${target}.tmp_extract"
    rm -rf "${tmp}"
    mkdir -p "${tmp}" "${target}"
    unzip -q -o "${archive}" -d "${tmp}"
    if [[ -d "${tmp}/VCTK-Corpus-0.92" ]]; then
        shopt -s dotglob nullglob
        mv "${tmp}/VCTK-Corpus-0.92"/* "${target}/"
        shopt -u dotglob nullglob
    else
        shopt -s dotglob nullglob
        mv "${tmp}"/* "${target}/"
        shopt -u dotglob nullglob
    fi
    rm -rf "${tmp}"
}

download_ears() {
    local target="${DATA_DIR}/EARS"
    local archive_dir="${target}/_archives"
    local urls_file="${archive_dir}/ears_urls.txt"
    mkdir -p "${target}" "${archive_dir}"

    python3 - "${urls_file}" <<'PY'
import json
import re
import sys
import urllib.request

out = sys.argv[1]
urls = []
try:
    req = urllib.request.Request(
        'https://api.github.com/repos/facebookresearch/ears_dataset/releases',
        headers={'User-Agent': 'addse-dataset-downloader'},
    )
    data = json.load(urllib.request.urlopen(req, timeout=60))
    for release in data:
        for asset in release.get('assets', []):
            url = asset.get('browser_download_url', '')
            name = asset.get('name', '')
            if re.search(r'p[0-9]{3}\.zip$', name) or re.search(r'p[0-9]{3}\.zip$', url):
                urls.append(url)
except Exception as exc:
    print(f'WARN EARS GitHub API lookup failed: {exc}', file=sys.stderr)

if not urls:
    urls = [f'https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{i:03d}.zip' for i in range(1, 108)]

with open(out, 'w') as fh:
    for url in sorted(set(urls)):
        fh.write(url + '\n')
PY

    while IFS= read -r url; do
        [[ -n "${url}" ]] || continue
        local file="${archive_dir}/$(basename "${url}")"
        local speaker="${file##*/}"
        speaker="${speaker%.zip}"
        if [[ -d "${target}/${speaker}" ]]; then
            echo "SKIP EARS ${speaker}: already extracted"
            continue
        fi
        download_file "${url}" "${file}"
        extract_zip_once "${file}" "${target}"
    done < "${urls_file}"
}

download_wham() {
    local target="${DATA_DIR}/WHAM_48kHz"
    local archive="${target}/high_res_wham.zip"
    mkdir -p "${target}"
    download_file "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip" "${archive}"
    if find "${target}" -type f -name '*.wav' -print -quit | grep -q .; then
        echo "SKIP WHAM_48kHz extraction: wav files already present"
    else
        extract_zip_once "${archive}" "${target}"
    fi
}

download_demand() {
    local target="${DATA_DIR}/DEMAND"
    local archive_dir="${target}/_archives"
    local urls_file="${archive_dir}/demand_urls.txt"
    mkdir -p "${target}" "${archive_dir}"

    if find "${target}" -type f -name '*.wav' -print -quit | grep -q .; then
        echo "SKIP DEMAND: wav files already present"
        return 0
    fi

    python3 - "${urls_file}" <<'PY'
import json
import re
import sys
import urllib.request

out = sys.argv[1]
data = json.load(urllib.request.urlopen('https://zenodo.org/api/records/1227121', timeout=60))
with open(out, 'w') as fh:
    for item in data['files']:
        key = item['key']
        if re.search(r'\.zip$', key):
            fh.write(f"{key}\t{item['links']['self']}\n")
PY

    while IFS=$'\t' read -r key url; do
        [[ -n "${key}" && -n "${url}" ]] || continue
        download_file "${url}" "${archive_dir}/${key}"
    done < "${urls_file}"

    if find "${target}" -type f -name '*.wav' -print -quit | grep -q .; then
        echo "SKIP DEMAND extraction: wav files already present"
    else
        for archive in "${archive_dir}"/*.zip; do
            [[ -e "${archive}" ]] || continue
            extract_zip_once "${archive}" "${target}"
        done
    fi
}

download_ears
download_wham
download_vctk
download_demand

echo "==== ADDSE four-dataset download finished: $(date -Is) ===="
