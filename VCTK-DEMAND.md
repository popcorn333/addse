Task: use only VCTK speech and DEMAND noise for training, validation, and test/evaluation. Do not use EARS or WHAM_48kHz for this task, even though they are available locally.

Important constraints:

* Work only inside the `addse` repository.
* Do not download any datasets.
* Do not delete original datasets.
* Do not modify files outside this repository.
* Do not train the model yet.
* Create deterministic train/test dataset chunks for VCTK + DEMAND.
* Use VCTK as clean speech.
* Use DEMAND as noise.
* Create a new config instead of overwriting the original `configs/addse-m.yaml`.

Available local datasets:

* `data/external/VCTK/`
* `data/external/DEMAND/`
* `data/external/EARS/`
* `data/external/WHAM_48kHz/`

Only use:

* `data/external/VCTK/`
* `data/external/DEMAND/`

Implementation requirements:

1. Inspect the repo’s data loading code and `addse ldopt` command.

   * Check `addse/data.py`.
   * Check the CLI implementation for `ldopt`.
   * Run `uv run addse ldopt --help` if needed.

2. Create a deterministic VCTK/DEMAND split-preparation script, for example:

   `scripts/prepare_vctk_demand.sh`

   The script should:

   * Find VCTK audio files under `data/external/VCTK/`.

   * Find DEMAND `.wav` files under `data/external/DEMAND/`.

   * Create train/test split directories or filelists under something like:

     `data/splits/vctk-demand/train/speech/`
     `data/splits/vctk-demand/test/speech/`
     `data/splits/vctk-demand/train/noise/`
     `data/splits/vctk-demand/test/noise/`

   * Prefer symlinks instead of copying audio files.

   * Use a fixed seed, e.g. `42`.

   * Prefer a speaker-disjoint split for VCTK if speaker IDs can be extracted from paths, for example `pXXX`.

   * Use a deterministic split for DEMAND, e.g. by sorted file paths, with about 90% train and 10% test.

   * Print the number of speech/noise files in each split.

3. Use `uv run addse ldopt` to create four LitData chunk folders:

   `data/chunks/vctk-demand/train_speech/`
   `data/chunks/vctk-demand/train_noise/`
   `data/chunks/vctk-demand/test_speech/`
   `data/chunks/vctk-demand/test_noise/`

   Use the correct regex based on actual VCTK files:

   * If VCTK files are `.flac`, use `--regexes "^.*\.flac$"`.
   * If VCTK files are `.wav`, use `--regexes "^.*\.wav$"`.

   DEMAND should use:

   * `--regexes "^.*\.wav$"`
   * `--labels demand`
   * `--seglens 10.0`

   Example commands to adapt after checking paths:

   ```bash
   uv run addse ldopt \
     data/splits/vctk-demand/train/speech/ \
     --regexes "^.*\.flac$" \
     --labels vctk \
     data/chunks/vctk-demand/train_speech/ \
     --num-workers 4 \
     --seed 42

   uv run addse ldopt \
     data/splits/vctk-demand/train/noise/ \
     --regexes "^.*\.wav$" \
     --labels demand \
     --seglens 10.0 \
     data/chunks/vctk-demand/train_noise/ \
     --num-workers 4 \
     --seed 42

   uv run addse ldopt \
     data/splits/vctk-demand/test/speech/ \
     --regexes "^.*\.flac$" \
     --labels vctk \
     data/chunks/vctk-demand/test_speech/ \
     --num-workers 4 \
     --seed 42

   uv run addse ldopt \
     data/splits/vctk-demand/test/noise/ \
     --regexes "^.*\.wav$" \
     --labels demand \
     --seglens 10.0 \
     data/chunks/vctk-demand/test_noise/ \
     --num-workers 4 \
     --seed 42
   ```

4. Create a new config:

   `configs/addse-m-vctk-demand.yaml`

   Copy it from `configs/addse-m.yaml`, then modify it as follows:

   Training:

   * `dm.train_dataset.speech_dataset.input_dir` should be:
     `data/chunks/vctk-demand/train_speech/`
   * `dm.train_dataset.noise_dataset.input_dir` should be:
     `data/chunks/vctk-demand/train_noise/`

   Validation:

   * Use the test split, not Hugging Face EARS/WHAM.
   * `dm.val_dataset.speech_dataset.input_dir` should be:
     `data/chunks/vctk-demand/test_speech/`
   * `dm.val_dataset.noise_dataset.input_dir` should be:
     `data/chunks/vctk-demand/test_noise/`

   Evaluation/test:

   * Replace the existing `eval.dsets` with a single dataset named `vctk-demand`.
   * The eval speech dataset should use:
     `data/chunks/vctk-demand/test_speech/`
   * The eval noise dataset should use:
     `data/chunks/vctk-demand/test_noise/`
   * Keep the existing metrics block unchanged.

5. Add a short README note or script comment explaining how to run this setup.

6. Run sanity checks only, not full training:

   * Confirm all four chunk directories exist.
   * Confirm each chunk directory contains LitData output.
   * Confirm `configs/addse-m-vctk-demand.yaml` parses.
   * Confirm train, validation, and eval dataloaders instantiate.
   * Load one training batch and one validation/test batch.

7. Report back:

   * Exact VCTK path used.
   * Exact DEMAND path used.
   * Number of train/test speech files.
   * Number of train/test noise files.
   * Chunk folders created.
   * Config file created.
   * Any code files or scripts modified.
   * Final commands to run.

Expected final commands:

```bash
bash scripts/prepare_vctk_demand.sh
```

```bash
uv run addse train configs/addse-m-vctk-demand.yaml
```

```bash
uv run addse eval \
  configs/addse-m-vctk-demand.yaml \
  logs/addse-m-vctk-demand/checkpoints/last.ckpt \
  --num-consumers 4
```

