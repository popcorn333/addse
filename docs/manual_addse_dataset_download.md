# Manual ADDSE Dataset Download Instructions

The ADDSE conversion scripts expect each dataset at the paths below under
`data/external/`. The previous README links are landing pages or official
workflow pages rather than direct archive URLs, so these datasets need manual
download, login, license acceptance, token-based tooling, or dataset-specific
file selection before conversion.

## EARS

- Source link: https://github.com/facebookresearch/ears_dataset
- Expected destination: `data/external/EARS/`

## LibriSpeech

- Source link: https://www.openslr.org/12
- Expected destination: `data/external/LibriSpeech/`
- ADDSE uses the `train-clean-100` and `train-clean-360` subsets.

## DNS5 Clean Speech

- Source link: https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
- Expected destination: `data/external/DNS/datasets_fullband/clean_fullband/`

## MLS_URGENT_2025_track1

- Source link: https://huggingface.co/datasets/kohei0209/mls_hq_urgent_track1
- Expected destination: `data/external/MLS_URGENT_2025_track1/`

## WHAM_48kHz

- Source link: http://wham.whisper.ai/
- Expected destination: `data/external/WHAM_48kHz/`

## DEMAND

- Source link: https://zenodo.org/records/1227121
- Expected destination: `data/external/DEMAND/`

## FSD50K

- Source link: https://zenodo.org/records/4060432
- Expected destination: `data/external/FSD50K/`
- ADDSE uses files matching `dev_audio`.

## DNS Noise

- Source link: https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
- Expected destination: `data/external/DNS/datasets_fullband/noise_fullband/`

## FMA_medium

- Source link: https://github.com/mdeff/fma
- Expected destination: `data/external/FMA_medium/`
