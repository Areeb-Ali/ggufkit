# ggufkit

Fingerprint, verify and visualize [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) models.

`ggufkit` answers questions a plain file hash cannot:

- Is this GGUF the same model as that one, even though the bytes differ?
- Did a re-upload change the weights, or only the metadata?
- Which tensors changed?
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

The first step is a content identity for GGUF files that survives the edits people
legitimately make. That is `ggufkit fingerprint`: separate, order-independent SHA-256
digests for the metadata, the tensor descriptors and the weights, on a documented byte
format ([docs/FINGERPRINT.md](docs/FINGERPRINT.md)). Manifests, verification and numeric
diffs build on it (see [Roadmap](#roadmap)).

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
$ ggufkit fingerprint Llama-3.2-1B-Q4_K_M.gguf
TENSOR   3c0f8a…  blk.0.attn_k.weight [Q4_K 2048x512]
TENSOR   91be02…  blk.0.attn_norm.weight [F32 2048]
…
kv           d07bb738a7971e4986babf859eb350353f1ed2672963076f0cf6081cf0641b04
tensor_info  02c3cc66758bee54878ad0d685409e64aaf44209159c323a0e360f5a55624f64
weights      c647b9b27a8ca782a8217b596c72b097551fa2dac8cbdcf373d5e93d7d274eef
overall      420d1bd00ea1dbb6fdc63c7b8e27e832db304314753882d0c1132ea8cf297b61
spec         ggufkit-fp/1  (26 metadata entries, 147 tensors)
```

Three digests, three answers:

| `kv` | `tensor_info` | `weights` | meaning |
|---|---|---|---|
| same | same | same | the same model |
| differs | same | same | only metadata changed (names, templates, licence text) |
| same | differs | same | tensors renamed or retyped in place, bytes untouched |
| any | any | differs | at least one tensor's bytes changed; the per-tensor lines say which |

Useful flags:

- `--json` prints the fingerprint as JSON; `-o fp.json` also writes it to a file.
- `--no-data` skips the tensor bytes: instant on any model size, gives you `kv` and
  `tensor_info` only.
- `--per-kv` adds a digest per metadata entry, so you can see exactly which key changed.
- `-j 8` hashes tensors on eight threads (reads are memory-mapped and never copied).

Pass any shard of a split model and the whole model is fingerprinted. Splitting or merging
does not change the result, and neither does llama.cpp's `gguf_new_metadata` (only `kv`
moves).

### Stitch metadata onto tensors

```sh
ggufkit stitch --metadata tokenizer-fixed.gguf --tensors model.gguf -o model-fixed.gguf
ggufkit stitch --metadata model.gguf -o vocab-only.gguf     # metadata only, no tensors
```

The result takes every metadata entry from `--metadata` and every tensor from `--tensors`.
Vocab-only files are handy as a base for training or further stitching.

### Look at the weights

```sh
ggufkit viz tensor model.gguf blk.0.attn_q.weight -o attn_q.png
ggufkit viz tensor model.gguf 'blk.*.attn_q.weight' --match-glob -o imgs/q.png
ggufkit viz tensor model.gguf token_embd.weight -o embd.png --max-pixels 4000000
```

Each pixel is one weight: brightness follows magnitude, hue says whether the weight sits
well below the mean, well above it, or in between. Attention projections have a lot going
on. A slice of an `attn_q` tensor from a 3B model:

![attn_q tensor rendered as an image](assets/attn_q_snippet.png)

PyTorch checkpoints work too when `torch` is installed (`--model-type torch`).

## What a fingerprint proves, and what it does not

A fingerprint proves integrity and equivalence: that two files hold the same metadata,
descriptors and weights, or exactly which part differs. It does not prove that a quantized
file was faithfully derived from the full-precision model it names, and it does not prove a
model is benign. Those need the build inputs (source hash, importance matrix, llama.cpp
revision, quantize flags) and either a reproduction or a numeric comparison. That is what
the roadmap is for.

## Roadmap

- `ggufkit manifest`: a JSON provenance record (fingerprint, file hashes, everything the
  metadata says about origin and licence, the build inputs you supply, and a list of the
  provenance keys that are missing).
- `ggufkit verify`: a file against a manifest, or two files against each other, with a
  one-word verdict (`identical`, `metadata-only`, `tensor-info-only`, `weights-changed`).
- `ggufkit diff`: metadata diff, tensor-level hash diff, and `--numeric` per-tensor error
  statistics between a quant and its source.
- `ggufkit viz error` and `viz layers`: quantization-error heatmaps and a per-layer map of
  which tensor got which quantization type.
- in-toto statements for signing with `cosign` or `model_signing`.

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests build tiny GGUF files in-process; nothing is downloaded. `pytest --update-golden`
rewrites the golden fingerprint after a deliberate spec change (bump the spec id too).

## Origins

ggufkit started from [KerfuffleV2/gguf-tools](https://github.com/KerfuffleV2/gguf-tools)
(MIT, 2023): `gguf-checksum`, `gguf-frankenstein` and `gguf-tensor-to-image`. Those commits
are preserved in this repository's history. The checksum tool's idea, hashing metadata,
tensor descriptors and weights separately and independent of their order, is the seed of
the fingerprint specification.

## License

MIT. See [LICENSE](LICENSE).
