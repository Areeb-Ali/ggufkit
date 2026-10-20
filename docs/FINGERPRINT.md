# ggufkit-fp/1: content fingerprints for GGUF models

Status: draft, implemented by `ggufkit.canon` and `ggufkit.fingerprint`. The spec id
`ggufkit-fp/1` is part of the hash input; any incompatible change gets a new id.

## Why a file hash is not enough

Two GGUF files that hold the same model routinely differ byte for byte:

- metadata was added, edited or reordered (`gguf_set_metadata`, `gguf_new_metadata`, a Hub
  re-upload with a fixed chat template);
- the model was split into shards with `gguf-split`, or merged back;
- tensors were written in a different order, or the alignment changed;
- the file was converted between little- and big-endian.

A signature over the raw bytes (what OpenSSF model-signing and `sha256sum` give you) breaks
on every one of those, and when it breaks it cannot tell you whether the weights changed. A
fingerprint can.

## What is hashed

All hashes are SHA-256. All integers are fixed-width little-endian. `‖` is concatenation,
`u32(n)` and `u64(n)` are 4- and 8-byte encodings, `len(x)` is a byte length.

### Metadata entry: `E_kv`

For every metadata key/value pair:

```
E_kv = sha256( "kv\0" ‖ u64(len(name)) ‖ name ‖ u32(vtype) ‖ value )
```

`name` is the key in UTF-8 and `vtype` is the GGUF value type id. `value` is encoded by type:

| type | encoding |
|---|---|
| UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64 | the integer, fixed width, little-endian |
| FLOAT32, FLOAT64 | the IEEE 754 bit pattern, little-endian (so `-0.0` and NaN payloads are preserved) |
| BOOL | one byte, `0x00` or `0x01` |
| STRING | `u64(len(s)) ‖ s` with `s` in UTF-8 |
| ARRAY | `u32(item_type) ‖ u64(count) ‖ item_1 ‖ … ‖ item_count`, items encoded with this table (nested arrays recurse) |

Excluded from `kv` entirely:

- the header counts (`GGUF.version`, `GGUF.tensor_count`, `GGUF.kv_count`), which describe
  the file, not the model;
- `split.no`, `split.count` and `split.tensors.count`, which describe sharding;
- placeholder entries whose name starts with `__SPARE`.

Everything else is included, `general.uuid` and `general.alignment` among them: they are
claims made by whoever wrote the file, and a verifier should see when they change.

### `kv`

```
kv = sha256( E_kv(name_1) ‖ E_kv(name_2) ‖ … )   for names in byte-wise UTF-8 order
```

### Tensor descriptor: `E_ti`

For every tensor:

```
E_ti = sha256( "ti\0" ‖ u64(len(name)) ‖ name ‖ u32(n_dims) ‖ u64(ne_0) ‖ … ‖ u64(ne_{n-1}) ‖ u32(ggml_type) )
```

Dimensions are in GGUF order (`ne[0]` varies fastest), exactly as stored. The file offset
is not included: it changes whenever anything before the tensor changes and says nothing
about the tensor.

### `tensor_info`

```
tensor_info = sha256( E_ti(name_1) ‖ E_ti(name_2) ‖ … )   for names in byte-wise UTF-8 order
```

### Tensor data: `D_t`

```
D_t = sha256( stored bytes of the tensor )
```

For element types (F32, F16, F64, I8, I16, I32, I64) in a big-endian file, values are
byte-swapped to little-endian before hashing, so the fingerprint does not depend on byte
order. Quantized types mix scales and packed values inside each block; their byte order
cannot be normalized without decoding the block, so quantized tensors in big-endian files
are refused in this version (convert with llama.cpp's `gguf_convert_endian` first).

Alignment padding between tensors is never hashed.

### `weights`

```
weights = sha256( D_1 ‖ D_2 ‖ … )   for digests sorted by their byte value
```

Sorting by digest rather than by name makes `weights` a hash of the *multiset* of tensor
contents: renaming or reordering tensors leaves it unchanged. A rename still shows, in
`tensor_info`.

### `overall`

```
overall = sha256( "ggufkit-fp/1\0" ‖ kv ‖ tensor_info ‖ weights )
```

## Sharded models

A model split with `gguf-split` is fingerprinted as one logical model: the metadata of all
shards is merged (first shard wins on duplicate keys, `split.*` keys dropped), and the
tensors of all shards are concatenated. The split and the merged file therefore have the
same fingerprint.

## Reading a comparison

| `kv` | `tensor_info` | `weights` | meaning |
|---|---|---|---|
| same | same | same | the same model (`overall` is equal too) |
| differs | same | same | only metadata changed: names, templates, licence text, provenance keys |
| same | differs | same | tensors were renamed or retyped in place, bytes untouched |
| any | any | differs | at least one tensor's bytes changed; per-tensor `data` digests say which |

## What a fingerprint does not prove

A fingerprint proves integrity and equivalence, nothing more. It does not prove that a
quantized file was faithfully derived from the full-precision model it names: that needs
the build inputs (source hash, importance matrix, llama.cpp revision, quantize flags) and
either a reproduction or a numeric comparison. ggufkit's manifest and `diff --numeric`
commands exist for that; a fingerprint is the thing they agree on.

## JSON form

```json
{
  "spec": "ggufkit-fp/1",
  "kv": "<hex>",
  "tensor_info": "<hex>",
  "weights": "<hex or null when --no-data>",
  "overall": "<hex or null when --no-data>",
  "n_kv": 23,
  "n_tensors": 291,
  "kvs": { "general.architecture": "<hex>", "...": "..." },
  "tensors": {
    "token_embd.weight": { "type": "Q4_K", "shape": [4096, 32000], "info": "<hex>", "data": "<hex>" }
  }
}
```

`kvs` is present only with `--per-kv`; `tensors` is absent with `--no-per-tensor`.
