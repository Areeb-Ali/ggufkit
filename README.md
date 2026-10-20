# ggufkit

Fingerprint, verify and visualize [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) models.

`ggufkit` answers questions a plain file hash cannot:

- Is this GGUF the same model as that one, even though the bytes differ?
- Did a re-upload change the weights, or only the metadata? Which keys, which tensors?
- Does this file document where it came from well enough to reproduce it?
- How far is this quantization from its source, tensor by tensor?
- What does `blk.17.attn_q.weight` actually look like?

It is built on llama.cpp's own [`gguf`](https://pypi.org/project/gguf/) Python package, so
every tensor type that package can decode works here too: 26 of them, from `Q4_0` to
`IQ2_XXS` and `MXFP4`. Sharded models (`-00001-of-00003.gguf`) are handled as one model.

## Why

Model signing ([OpenSSF model-signing](https://github.com/sigstore/model-transparency),
Sigstore) protects the bytes you were given. It breaks the moment someone fixes a chat
template in the metadata, and when it breaks it cannot tell you whether the weights changed.
It also says nothing about whether a community quantization was faithfully built from the
model it names: a 2026 study of the most-downloaded GGUF repositories on Hugging Face found
that none record enough build inputs to reproduce the quantization, and
[Mind the Gap](https://arxiv.org/abs/2505.23786) (ICML 2025) showed weights that behave
well in full precision and badly after GGUF quantization.

ggufkit gives GGUF files a content identity that survives the edits people legitimately make
([docs/FINGERPRINT.md](docs/FINGERPRINT.md)), a provenance record that says what a file claims
and what it leaves out ([docs/MANIFEST.md](docs/MANIFEST.md)), and the tools to check one
against the other.

## Install

```sh
pip install 'ggufkit[viz] @ git+https://github.com/MysterionRise/ggufkit'   # until the first PyPI release
uv tool install 'ggufkit[viz] @ git+https://github.com/MysterionRise/ggufkit'
```

The `viz` extra pulls in Pillow for image rendering; the core needs only `gguf` and `numpy`.
Python 3.10 or newer.

## Quick start

### Fingerprint a model

```console
$ ggufkit fingerprint model.gguf
TENSOR   3c0f8a…  blk.0.attn_k.weight [Q4_K 2048x512]
TENSOR   91be02…  blk.0.attn_norm.weight [F32 2048]
…
kv           d07bb738…
tensor_info  02c3cc66…
weights      c647b9b2…
overall      420d1bd0…
spec         ggufkit-fp/1  (26 metadata entries, 147 tensors)
```

Three digests, three answers:

| `kv` | `tensor_info` | `weights` | meaning |
|---|---|---|---|
| same | same | same | the same model |
| differs | same | same | only metadata changed (names, templates, licence text) |
| same | differs | same | tensors renamed or retyped in place, bytes untouched |
| any | any | differs | tensor bytes changed, or tensors were added or removed |

`--json` prints JSON, `--no-data` skips tensor bytes (instant on any size), `--per-kv` adds a
digest per metadata entry, `-j 8` hashes on eight threads. Pass any shard of a split model and
the whole model is fingerprinted; splitting, merging or editing metadata with llama.cpp's
`gguf_new_metadata` leaves `weights` unchanged.

### Record provenance

```console
$ ggufkit manifest model-Q4_K_M.gguf --input llama_cpp_commit=b8146
manifest     model-Q4_K_M.manifest.json
model        llama Q4_K_M, 147 tensors, 1.24B parameters, 762.81 MiB
fingerprint  420d1bd0…
provenance   license, quantized_by
score        3/9 of what a verifiable file records
gaps         general.source.*, general.uuid, imatrix.datasets, build_inputs.source_sha256, build_inputs.imatrix_sha256, build_inputs.quantize_cmd
```

The manifest is a JSON file: the full fingerprint, file sizes (and raw hashes with
`--file-sha256`), what the metadata claims about origin and licence, the build inputs you pass
with `--input KEY=VALUE`, and the list of what a reproducible quantization would record but
this file does not. Publish it next to the model.

### Verify

```console
$ ggufkit verify model-Q4_K_M.gguf --manifest model-Q4_K_M.manifest.json
verdict      metadata-only
kv           differs
tensor_info  same
weights      same
metadata changes:
  ~ general.name
```

Verify against a manifest or straight against another file with `--against other.gguf`. The
verdict is one of `identical`, `metadata-only`, `tensor-info-only`, `metadata-and-tensor-info`
or `weights-changed`. The exit status is 0 only for `identical`, so it drops into CI and
download scripts; `--allow-metadata` also accepts `metadata-only`, and `--ignore-keys` takes
globs for keys you expect to change.

### Diff two models

```console
$ ggufkit diff model-F32.gguf model-Q4_0.gguf --numeric
metadata: 0 changed, 0 added, 0 removed (of 11)
tensors: 2 changed, 0 added, 0 removed (of 3)
  ~ blk.0.attn_q.weight  type, data  [F32 256x64 -> Q4_0 256x64]
  ~ blk.0.ffn_up.weight  type, data  [F32 256x32 -> Q4_0 256x32]
numeric (top 3 by rel_l2):
  tensor                               a -> b                 rmse    max_abs   mean_abs     rel_l2     cosine
  blk.0.attn_q.weight                  F32->Q4_0          0.019035   0.062021   0.015899   0.065937   0.997835
  blk.0.ffn_up.weight                  F32->Q4_0          0.019037   0.060780   0.015898   0.065937   0.997836
  output_norm.weight                   F32->F32           0.000000   0.000000   0.000000   0.000000   1.000000
```

Metadata entries are shown with their old and new values; tensors are listed as added,
removed or changed (shape, type, data). `--numeric` decodes both sides and reports error per
tensor, worst first, in bounded memory. `--tensor 'blk.*.attn_*'` narrows it, `--json` gives
machine-readable output, and the exit status is 1 when anything differs.

### Stitch metadata onto tensors

```sh
ggufkit stitch --metadata tokenizer-fixed.gguf --tensors model.gguf -o model-fixed.gguf
ggufkit stitch --metadata model.gguf -o vocab-only.gguf     # metadata only, no tensors
```

### Look at the weights

```sh
ggufkit viz tensor model.gguf blk.0.attn_q.weight -o attn_q.png
ggufkit viz tensor model.gguf 'blk.*.attn_q.weight' --match-glob -o imgs/q.png
ggufkit viz tensor model.gguf token_embd.weight -o embd.png --max-pixels 4000000
```

Each pixel is one weight: brightness follows magnitude, hue says whether the weight sits
well below the mean, well above it, or in between. A slice of an `attn_q` tensor from a 3B
model:

![attn_q tensor rendered as an image](assets/attn_q_snippet.png)

PyTorch checkpoints work too when `torch` is installed (`--model-type torch`).

## What this proves, and what it does not

A fingerprint proves integrity and equivalence: that two files hold the same metadata,
descriptors and weights, or exactly which part differs. A manifest records claims and makes
their absence visible. Neither proves that a quantized file was faithfully derived from the
model it names, or that a model is benign. For that, re-quantize the source with the recorded
build inputs and compare: `ggufkit verify --against` tells you whether you reproduced the
file, and `ggufkit diff --numeric` tells you how far apart the two are when you did not.

## Roadmap

- `ggufkit viz error` and `viz layers`: quantization-error heatmaps and a per-layer map of
  which tensor got which quantization type.
- in-toto statements from manifests, for signing with `cosign` or `model_signing`.
- A one-shot "reproduce this quant" check that drives `llama-quantize` from a manifest.

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests build tiny GGUF files in-process; nothing is downloaded. `pytest --update-golden`
rewrites the golden fingerprint after a deliberate spec change (bump the spec id too).
Releases are published to PyPI by pushing a `vX.Y.Z` tag that matches `pyproject.toml`.

## Origins

ggufkit started from [KerfuffleV2/gguf-tools](https://github.com/KerfuffleV2/gguf-tools)
(MIT, 2023): `gguf-checksum`, `gguf-frankenstein` and `gguf-tensor-to-image`. Those commits
are preserved in this repository's history. The checksum tool's idea, hashing metadata,
tensor descriptors and weights separately and independent of their order, is the seed of
the fingerprint specification.

## License

MIT. See [LICENSE](LICENSE).
