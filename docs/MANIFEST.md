# ggufkit-manifest/1: provenance records for GGUF models

Status: draft, implemented by `ggufkit.manifest`. Readers must check `schema` and refuse
versions they do not know.

## Why

A 2026 study of the most-downloaded GGUF repositories on Hugging Face found that none record
enough to reproduce how the file was quantized. A manifest is the record that is missing: what
the file *is* (its fingerprint), what its metadata *claims* (origin, licence, who quantized
it), what the producer *declares* (build inputs), and an honest list of what is absent.

```sh
ggufkit manifest model-Q4_K_M.gguf \
  --input source_sha256=<sha256 of the BF16 source> \
  --input imatrix_sha256=<sha256 of imatrix.dat> \
  --input llama_cpp_commit=b8146 \
  --input "quantize_cmd=llama-quantize --imatrix imatrix.dat model-BF16.gguf model-Q4_K_M.gguf Q4_K_M" \
  --file-sha256
```

This writes `model-Q4_K_M.manifest.json` next to the model. Publish it with the model; anyone
can then run `ggufkit verify model-Q4_K_M.gguf --manifest model-Q4_K_M.manifest.json`.

## Fields

| field | content |
|---|---|
| `schema` | `"ggufkit-manifest/1"` |
| `created_at` | UTC timestamp, ISO 8601 |
| `tool` | `name`, `version`, `gguf_py` (version of llama.cpp's Python package), `python`, `fingerprint_spec` |
| `files[]` | one entry per shard: `path` (file name), `size`, `sha256` (raw bytes, `null` unless `--file-sha256`), `shard` (`{no, count}` or `null`) |
| `gguf` | `version`, `endianness`, `alignment`, `n_kv`, `n_tensors`, `n_shards` |
| `model` | `architecture`, `file_type` (e.g. `Q4_K_M`, as claimed by `general.file_type`), `file_type_id`, `quantization_version`, `parameter_count`, `tensor_bytes`, `tensor_type_counts`, `tensor_type_bytes` |
| `provenance` | what the metadata says; only keys that are present: `uuid`, `name`, `author`, `version`, `organization`, `basename`, `finetune`, `size_label`, `description`, `quantized_by`, `url`, `doi`, `repo_url`, `tags`, `languages`, `license {id, name, link}`, `source {url, doi, uuid, repo_url}`, `base_models[]`, `datasets[]`, `imatrix {chunk_count, chunk_size, datasets}` |
| `provenance_gaps` | ids from the table below that this file does not satisfy |
| `provenance_score` | `{recorded, expected}` |
| `build_inputs` | the `--input KEY=VALUE` pairs, verbatim |
| `notes` | free text from `--note`, or `null` |
| `fingerprint` | the full `ggufkit-fp/1` fingerprint with per-key and per-tensor digests (see [FINGERPRINT.md](FINGERPRINT.md)) |

## What a verifiable file records

| id | satisfied when |
|---|---|
| `general.source.*` | any of `general.source.url/doi/uuid/repo_url`, or at least one `general.base_model.*` entry |
| `general.quantized_by` | present and non-empty |
| `general.license` | any of `general.license`, `general.license.name`, `general.license.link` |
| `general.uuid` | present |
| `imatrix.datasets` | present; only expected when some tensor is stored below 8 bits |
| `build_inputs.source_sha256` | supplied with `--input` |
| `build_inputs.imatrix_sha256` | supplied with `--input`; only expected below 8 bits |
| `build_inputs.llama_cpp_commit` | supplied with `--input` |
| `build_inputs.quantize_cmd` | supplied with `--input` |

`expected` is 9 for a file with sub-8-bit tensors and 7 otherwise. The score measures
documentation, not trustworthiness: every value except the fingerprint is a claim.

## Verifying

```sh
ggufkit verify model.gguf --manifest model.manifest.json            # fingerprint only
ggufkit verify model.gguf --manifest model.manifest.json --file-sha256  # also raw file bytes
ggufkit verify model.gguf --against other.gguf                       # two files, no manifest
```

| verdict | meaning |
|---|---|
| `identical` | metadata, tensor descriptors and weights all match |
| `metadata-only` | only metadata differs; the report lists the keys |
| `tensor-info-only` | only tensor descriptors differ (renamed or retyped in place, same bytes) |
| `metadata-and-tensor-info` | both of the above, weights untouched |
| `weights-changed` | the bytes of at least one tensor differ, or tensors were added or removed |

Exit status: `0` for `identical` (and for `metadata-only` with `--allow-metadata`), `1` for any
other verdict or a raw file hash mismatch, `2` for errors. `--ignore-keys 'general.name,tokenizer.chat_template'`
excludes keys you expect to change. A manifest made with `--no-data` carries no weights digest,
so verification against it says `weights: not compared`.
