# Contributing to ggufkit

Thanks for helping. Bug reports with a reproducing command are as welcome as code.

## Set up

```sh
git clone https://github.com/MysterionRise/ggufkit
cd ggufkit
uv sync --all-extras
```

[uv](https://docs.astral.sh/uv/) manages the virtual environment and the lockfile. Python 3.10
or newer.

## Before you open a pull request

Run what CI runs:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

CI repeats these on Ubuntu and Windows for Python 3.10 to 3.13 and builds the wheel.

## Tests

Tests build tiny GGUF files in-process with `GGUFWriter` from deterministic data, so they need
no downloads and run in seconds. Add a test with every behaviour change; `tests/conftest.py`
has `write_gguf()` and fixtures for the common cases.

## The fingerprint and manifest formats are specifications

`docs/FINGERPRINT.md` and `docs/MANIFEST.md` describe byte formats other people may implement
and depend on.

- `tests/test_golden.py` pins the `ggufkit-fp/1` byte format. If your change makes it fail, you
  have changed the specification: give it a new id (`ggufkit-fp/2`), update the docs, and only
  then run `uv run pytest --update-golden`.
- Manifest changes that remove or rename fields need a new schema id; adding optional fields
  does not.

## README images and examples

The README's images and console output come from a real model. To regenerate them (about 2 GB
of downloads from Hugging Face, pinned to one revision and checked by sha256):

```sh
uv run python scripts/make_readme_assets.py
```

`scripts/make_examples.py` regenerates the one synthetic illustration. Paste console output
verbatim; do not edit numbers by hand.

## Style

- ruff's settings in `pyproject.toml` are the style guide; the formatter has the last word.
- Keep runtime dependencies to `gguf` and `numpy`. Pillow is the `viz` extra; anything heavier
  needs a good reason.
- Commit messages explain why, not only what.

## Licence

By contributing you agree that your contribution is licensed under the project's MIT licence.
