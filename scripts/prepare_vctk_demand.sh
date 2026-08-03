#!/usr/bin/env bash
set -euo pipefail

# Prepare deterministic, symlinked VCTK + DEMAND splits for ADD-SE.
# Run from the repository root:
#   bash scripts/prepare_vctk_demand.sh

python3 - <<'PY'
from __future__ import annotations

import hashlib
import random
import re
import shutil
from pathlib import Path

SEED = 42
TRAIN_FRACTION = 0.9

repo = Path.cwd()
vctk_root = repo / "data" / "external" / "VCTK"
demand_root = repo / "data" / "external" / "DEMAND"
split_root = repo / "data" / "splits" / "vctk-demand"

if not vctk_root.is_dir():
    raise SystemExit(f"Missing VCTK directory: {vctk_root}")
if not demand_root.is_dir():
    raise SystemExit(f"Missing DEMAND directory: {demand_root}")

speech_files = sorted(
    p for ext in ("*.flac", "*.wav") for p in vctk_root.rglob(ext) if p.is_file()
)
noise_files = sorted(p for p in demand_root.rglob("*.wav") if p.is_file())

if not speech_files:
    raise SystemExit(f"Found no VCTK .flac/.wav files under {vctk_root}")
if not noise_files:
    raise SystemExit(f"Found no DEMAND .wav files under {demand_root}")

speaker_re = re.compile(r"^p\d{3}$")
speakers: dict[str, list[Path]] = {}
unassigned: list[Path] = []
for path in speech_files:
    speaker = next((part for part in path.parts if speaker_re.match(part)), None)
    if speaker is None:
        unassigned.append(path)
    else:
        speakers.setdefault(speaker, []).append(path)

if speakers:
    speaker_ids = sorted(speakers)
    rng = random.Random(SEED)
    rng.shuffle(speaker_ids)
    test_count = max(1, round(len(speaker_ids) * (1 - TRAIN_FRACTION)))
    test_speakers = set(sorted(speaker_ids[:test_count]))
    train_speech = [p for speaker in sorted(set(speaker_ids) - test_speakers) for p in speakers[speaker]]
    test_speech = [p for speaker in sorted(test_speakers) for p in speakers[speaker]]
    if unassigned:
        # Keep any files without speaker IDs deterministic without mixing them into the speaker-disjoint VCTK split.
        train_speech.extend(unassigned)
else:
    rng = random.Random(SEED)
    shuffled = speech_files[:]
    rng.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * (1 - TRAIN_FRACTION)))
    train_speech = sorted(shuffled[test_count:])
    test_speech = sorted(shuffled[:test_count])
    test_speakers = set()

noise_cut = max(1, round(len(noise_files) * TRAIN_FRACTION))
train_noise = noise_files[:noise_cut]
test_noise = noise_files[noise_cut:]

if not test_noise:
    test_noise = train_noise[-1:]
    train_noise = train_noise[:-1]

if split_root.exists():
    shutil.rmtree(split_root)

def link_files(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for src in files:
        rel = src.relative_to(repo)
        digest = hashlib.sha1(str(rel).encode()).hexdigest()[:10]
        stem = "__".join(src.relative_to(src.parents[2] if "VCTK" in src.parts else demand_root).parts)
        name = f"{src.stem}_{digest}{src.suffix}" if stem in seen else stem
        seen.add(stem)
        dst = dst_dir / name
        dst.symlink_to(Path("../../../../../") / rel)

link_files(sorted(train_speech), split_root / "train" / "speech")
link_files(sorted(test_speech), split_root / "test" / "speech")
link_files(train_noise, split_root / "train" / "noise")
link_files(test_noise, split_root / "test" / "noise")

print(f"VCTK path: {vctk_root}")
print(f"DEMAND path: {demand_root}")
if speakers:
    print(f"VCTK speakers: {len(speakers)} total, {len(test_speakers)} test")
print(f"train speech files: {len(train_speech)}")
print(f"test speech files: {len(test_speech)}")
print(f"train noise files: {len(train_noise)}")
print(f"test noise files: {len(test_noise)}")
print(f"split root: {split_root}")
PY
