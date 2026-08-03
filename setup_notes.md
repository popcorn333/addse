# ADDSE Clean Setup Notes

## Repository and caches

The repository stays on the host at:

```bash
/mnt/parscratch/users/acp23xt/private/codex_addse/addse
```

The cache/temp base stays on the host at:

```bash
/mnt/parscratch/users/acp23xt/private/cache_temp
```

Do not clone the repository into `/`, `/root`, or into the Apptainer image. The image is only a runtime base; the host repository is bind-mounted when the container runs.

## Python version

ADDSE requires Python 3.12:

- `.python-version` contains `3.12`
- `pyproject.toml` requires `>=3.12,<3.13`
- `uv.lock` resolves `requires-python = "==3.12.*"`

## uv sync / onnxruntime diagnosis

`uv sync --dry-run` on the login node fails before installation with:

```text
Distribution `onnxruntime==1.23.2` can't be installed because it doesn't have a source distribution or wheel for the current platform
```

The login-node platform is treated by uv as `manylinux_2_17_x86_64`. The locked `onnxruntime==1.23.2` Linux wheels are only available for `manylinux_2_27` / `manylinux_2_28` on `x86_64` and `aarch64`.

The clean fix is to run inside a newer Linux userspace. `apptainer/addse.def` uses Ubuntu 24.04, which has a new enough glibc for the locked `onnxruntime` wheel tags.

## Build the Apptainer image

Build from the host repository:

```bash
cd /mnt/parscratch/users/acp23xt/private/codex_addse/addse
export APPTAINER_CACHEDIR=/mnt/parscratch/users/acp23xt/private/cache_temp/apptainer
export APPTAINER_TMPDIR=/mnt/parscratch/users/acp23xt/private/cache_temp/tmp
apptainer build addse.sif apptainer/addse.def
```

The definition file uses `/tmp` inside `%post`; it does not reference `/mnt/parscratch` during the image build.

## Runtime environment

Set these before running `uv` or Python inside the container:

```bash
export TMPDIR=/cache_temp/tmp
export UV_CACHE_DIR=/cache_temp/uv-cache
export HF_HOME=/cache_temp/hf-cache
export TORCH_HOME=/cache_temp/torch-cache
```

The Slurm script creates these cache directories and runs from:

```bash
/workspace/addse
```

## Slurm smoke test

Submit:

```bash
cd /mnt/parscratch/users/acp23xt/private/codex_addse/addse
sbatch slurm/test_addse_apptainer.sbatch
```

The script bind-mounts:

```bash
/mnt/parscratch/users/acp23xt/private/codex_addse/addse:/workspace/addse
/mnt/parscratch/users/acp23xt/private/cache_temp:/cache_temp
```

It runs only lightweight checks on the compute node:

```bash
python -c "import sys; print(sys.version)"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import addse; print('addse import OK')"
addse --help
```

No training or large data preparation is run on the login node.
