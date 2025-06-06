# SMAI Demo

This demonstrates our pipeline for processing chess games.

Inference pipeline:

picture -> corners model -> warp transform ->  piece detection

## How to use

You need the UV package/project manager to install the dependencies.  
You can get it from [here](https://docs.astral.sh/uv/getting-started/installation/).

> [!NOTE]
> To change the Python version, change the `requires-python` field in [pyproject.toml](pyproject.toml)
> and the number in [.python-version](.python-version).  
> uv will take care of the rest.

Set up the environment. (Only once)

```bash
uv venv
# .venv/Scripts/activate # Windows
source .venv/bin/activate # Linux/MacOS
uv sync --link-mode=symlink # Install the dependencies, use -U to update
```

If you want Pytorch (with or without CUDA), you can install it using the `--extra` flag.

```bash
uv sync --link-mode=symlink --extra=torch-cpu   # for CPU only
uv sync --link-mode=symlink --extra=torch-cu124 # for CUDA support
```

You can add other dependencies use `uv add`.

```bash
uv add numpy # Similar to pip install numpy
```

To run any script, append `uv run` before the `python` command. (If the environment is inactive)

```bash
uv run python src/hello.py
```

### Running the Chess Analysis Pipeline

Run the pipeline with default settings:

```bash
python pipeline.py --image path/to/your/chess/image.jpg
```

Process a folder of chess board images:

```bash
python pipeline.py --folder path/to/folder/with/images
```

Additional command line arguments:

```bash
# Specify custom model paths
python pipeline.py --image path/to/image.jpg --corner-model path/to/corner/model.pt --piece-model path/to/piece/model.pt

# Disable plot display (useful for automation)
python pipeline.py --image path/to/image.jpg --no-display

# Process a folder with custom models
python pipeline.py --folder path/to/folder --corner-model path/to/model.pt --piece-model path/to/model.pt --no-display
```

Run tests:

```bash
uv run python -m unittest discover
```

Get rid of temporary files: (Use with caution)

```bash
git clean -fdX -n # Remove the -n flag to actually delete the files
```

## Code Formatting and Linting

We have [ruff](https://docs.astral.sh/ruff/) for code formatting and linting.
Install the [VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
and enable `Format on Save` for a better experience.

To sort imports:

```bash
uv run ruff check --select I --fix
```

To check for linting errors:

```bash
uv run ruff check # Use --fix to fix the errors
```

To format the code:

```bash
uv run ruff format
```

## Model training code and output

The model training code is in the jupyter notebooks in the `notebooks` folder.

Also see:

1. <https://iiithydstudents-my.sharepoint.com/personal/ankita_porel_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fankita%5Fporel%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2Fankita&ga=1>
