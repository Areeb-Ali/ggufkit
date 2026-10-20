# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

First release.

### Added

- `ggufkit fingerprint`: order-independent content fingerprint (`ggufkit-fp/1`, see
  `docs/FINGERPRINT.md`) with separate digests for metadata, tensor descriptors and weights.
  Invariant to metadata and tensor order, offsets, alignment, sharding and, for element types,
  byte order.
- `ggufkit manifest`: provenance record (`ggufkit-manifest/1`, see `docs/MANIFEST.md`) with the
  fingerprint, file hashes, metadata claims, user-supplied build inputs and a provenance gap list.
- `ggufkit verify`: compare a model with a manifest or another file and get a one-word verdict
  with a meaningful exit status.
- `ggufkit diff`: metadata value diff, tensor diff and `--numeric` per-tensor error statistics.
- `ggufkit stitch`: combine the metadata of one file with the tensors of another, or write a
  vocab-only file.
- `ggufkit viz tensor`: render tensors of any type `gguf.quants` can decode as images.
- Split models (`-00001-of-00003.gguf`) are handled as one model everywhere.

### Origins

Builds on [KerfuffleV2/gguf-tools](https://github.com/KerfuffleV2/gguf-tools) (MIT, 2023):
`gguf-checksum`, `gguf-frankenstein` and `gguf-tensor-to-image`, whose history is preserved.
