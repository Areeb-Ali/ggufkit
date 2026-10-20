"""Provenance manifests for GGUF models (``ggufkit-manifest/1``).

A manifest is a JSON record of everything ggufkit can establish about a file: its
fingerprint, the plain file hashes, what the GGUF header says, what the metadata claims about
the model's origin and licence, and the build inputs the person who made the file chose to
record. It also lists what is *missing*: the provenance keys and build inputs a verifiable
quantization would need and this file does not carry. Format details are in
``docs/MANIFEST.md``.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gguf import GGMLQuantizationType, GGUFEndian
from gguf.constants import Keys, LlamaFileType

from ggufkit import __version__, canon
from ggufkit.fingerprint import Fingerprint, default_jobs, fingerprint
from ggufkit.model import SHARD_RE, open_model

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

    from ggufkit.model import Model

SCHEMA = "ggufkit-manifest/1"

#: Build inputs a reproducible quantization needs. They are not in the GGUF metadata, so the
#: person who produced the file supplies them with ``--input``.
BUILD_INPUT_KEYS = ("source_sha256", "imatrix_sha256", "llama_cpp_commit", "quantize_cmd")

#: Tensor types that do not benefit from an importance matrix.
_HIGH_PRECISION = frozenset(
    {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.F64,
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q8_1,
        GGMLQuantizationType.Q8_K,
    }
)

#: What a verifiable file records, as (id, explanation). Order is the order in reports.
EXPECTED_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("general.source.*", "where the weights came from (source url, repo or uuid, or a base model)"),
    ("general.quantized_by", "who produced this file"),
    ("general.license", "the licence the file is distributed under"),
    ("general.uuid", "a stable identifier for the model"),
    ("imatrix.datasets", "the importance-matrix data used for sub-8-bit quantization"),
    ("build_inputs.source_sha256", "hash of the full-precision source the quant was made from"),
    ("build_inputs.imatrix_sha256", "hash of the importance matrix file"),
    ("build_inputs.llama_cpp_commit", "the llama.cpp revision that produced the file"),
    ("build_inputs.quantize_cmd", "the exact quantize command line"),
)


class ManifestError(ValueError):
    """Raised for manifests ggufkit cannot read or does not understand."""


def _kv(model: Model, key: str) -> Any | None:
    f = model.fields.get(key)
    if f is None or not f.types:
        return None
    value = f.contents()
    if value in ("", [], None):
        return None
    return value


def _indexed(model: Model, count_key: str, templates: Mapping[str, str]) -> list[dict[str, Any]]:
    count = _kv(model, count_key)
    if not count:
        return []
    out: list[dict[str, Any]] = []
    for i in range(int(count)):
        entry = {
            name: v
            for name, template in templates.items()
            if (v := _kv(model, template.format(id=i))) is not None
        }
        out.append(entry)
    return out


def extract_provenance(model: Model) -> dict[str, Any]:
    """Everything the metadata says about where the model came from and who made it."""
    g = Keys.General
    simple = {
        "uuid": g.UUID,
        "name": g.NAME,
        "author": g.AUTHOR,
        "version": g.VERSION,
        "organization": g.ORGANIZATION,
        "basename": g.BASENAME,
        "finetune": g.FINETUNE,
        "size_label": g.SIZE_LABEL,
        "description": g.DESCRIPTION,
        "quantized_by": g.QUANTIZED_BY,
        "url": g.URL,
        "doi": g.DOI,
        "repo_url": g.REPO_URL,
        "tags": g.TAGS,
        "languages": g.LANGUAGES,
    }
    prov: dict[str, Any] = {k: v for k, key in simple.items() if (v := _kv(model, key)) is not None}

    licence = {
        k: v
        for k, key in {"id": g.LICENSE, "name": g.LICENSE_NAME, "link": g.LICENSE_LINK}.items()
        if (v := _kv(model, key)) is not None
    }
    if licence:
        prov["license"] = licence

    source = {
        k: v
        for k, key in {
            "url": g.SOURCE_URL,
            "doi": g.SOURCE_DOI,
            "uuid": g.SOURCE_UUID,
            "repo_url": g.SOURCE_REPO_URL,
        }.items()
        if (v := _kv(model, key)) is not None
    }
    if source:
        prov["source"] = source

    entry_fields = {
        "name": "NAME",
        "author": "AUTHOR",
        "version": "VERSION",
        "organization": "ORGANIZATION",
        "description": "DESCRIPTION",
        "url": "URL",
        "doi": "DOI",
        "uuid": "UUID",
        "repo_url": "REPO_URL",
    }
    base_models = _indexed(
        model,
        g.BASE_MODEL_COUNT,
        {k: getattr(g, f"BASE_MODEL_{attr}") for k, attr in entry_fields.items()},
    )
    if base_models:
        prov["base_models"] = base_models
    datasets = _indexed(
        model,
        g.DATASET_COUNT,
        {k: getattr(g, f"DATASET_{attr}") for k, attr in entry_fields.items()},
    )
    if datasets:
        prov["datasets"] = datasets

    im = Keys.IMatrix
    imatrix = {
        k: v
        for k, key in {
            "chunk_count": im.CHUNK_COUNT,
            "chunk_size": im.CHUNK_SIZE,
            "datasets": im.DATASETS,
        }.items()
        if (v := _kv(model, key)) is not None
    }
    if imatrix:
        prov["imatrix"] = imatrix
    return prov


def needs_imatrix(model: Model) -> bool:
    """Return whether any tensor is stored below 8 bits, where an importance matrix matters."""
    return any(t.tensor_type not in _HIGH_PRECISION for t in model.tensors)


def provenance_gaps(
    provenance: Mapping[str, Any],
    build_inputs: Mapping[str, str],
    *,
    quantized: bool,
) -> list[str]:
    """Ids from :data:`EXPECTED_PROVENANCE` that this file does not satisfy."""
    gaps: list[str] = []
    has_source = bool(provenance.get("source")) or bool(provenance.get("base_models"))
    if not has_source:
        gaps.append("general.source.*")
    if not provenance.get("quantized_by"):
        gaps.append("general.quantized_by")
    if not provenance.get("license"):
        gaps.append("general.license")
    if not provenance.get("uuid"):
        gaps.append("general.uuid")
    if quantized and not (provenance.get("imatrix") or {}).get("datasets"):
        gaps.append("imatrix.datasets")
    for key in BUILD_INPUT_KEYS:
        if key == "imatrix_sha256" and not quantized:
            continue
        if not build_inputs.get(key):
            gaps.append(f"build_inputs.{key}")
    return gaps


def expected_count(*, quantized: bool) -> int:
    return len(EXPECTED_PROVENANCE) - (0 if quantized else 2)


def file_type_label(file_type_id: int | None) -> str | None:
    if file_type_id is None:
        return None
    try:
        name = LlamaFileType(int(file_type_id)).name
    except ValueError:
        return f"unknown({file_type_id})"
    for prefix in ("MOSTLY_", "ALL_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def model_facts(model: Model) -> dict[str, Any]:
    file_type_id = _kv(model, Keys.General.FILE_TYPE)
    counts = Counter(t.tensor_type.name for t in model.tensors)
    bytes_by_type: Counter[str] = Counter()
    for t in model.tensors:
        bytes_by_type[t.tensor_type.name] += int(t.n_bytes)
    return {
        "architecture": model.architecture,
        "file_type": file_type_label(file_type_id),
        "file_type_id": None if file_type_id is None else int(file_type_id),
        "quantization_version": _kv(model, Keys.General.QUANTIZATION_VERSION),
        "parameter_count": sum(int(t.n_elements) for t in model.tensors),
        "tensor_bytes": sum(int(t.n_bytes) for t in model.tensors),
        "tensor_type_counts": dict(counts.most_common()),
        "tensor_type_bytes": dict(bytes_by_type.most_common()),
    }


def sha256_file(path: Path, chunk_size: int = canon.CHUNK_SIZE) -> str:
    digest = sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_entries(model: Model, *, with_sha256: bool) -> list[dict[str, Any]]:
    n = model.n_shards
    return [
        {
            "path": p.name,
            "size": p.stat().st_size,
            "sha256": sha256_file(p) if with_sha256 else None,
            "shard": {"no": i, "count": n} if n > 1 else None,
        }
        for i, p in enumerate(model.paths, 1)
    ]


def parse_build_inputs(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            msg = f"--input expects KEY=VALUE, got {item!r}"
            raise ManifestError(msg)
        out[key.strip()] = value.strip()
    return out


def build_manifest(
    model: Model,
    *,
    fp: Fingerprint | None = None,
    file_sha256: bool = False,
    build_inputs: Mapping[str, str] | None = None,
    notes: str | None = None,
    data: bool = True,
    per_tensor: bool = True,
    jobs: int = 1,
) -> dict[str, Any]:
    if fp is None:
        fp = fingerprint(model, data=data, per_tensor=per_tensor, per_kv=True, jobs=jobs)
    inputs = dict(build_inputs or {})
    provenance = extract_provenance(model)
    quantized = needs_imatrix(model)
    gaps = provenance_gaps(provenance, inputs, quantized=quantized)
    try:
        gguf_py = version("gguf")
    except PackageNotFoundError:  # pragma: no cover
        gguf_py = "unknown"
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "name": "ggufkit",
            "version": __version__,
            "gguf_py": gguf_py,
            "python": platform.python_version(),
            "fingerprint_spec": canon.SPEC_ID,
        },
        "files": file_entries(model, with_sha256=file_sha256),
        "gguf": {
            "version": model.version,
            "endianness": "big" if model.endianess == GGUFEndian.BIG else "little",
            "alignment": model.alignment,
            "n_kv": len(model.metadata_fields()),
            "n_tensors": len(model.tensors),
            "n_shards": model.n_shards,
        },
        "model": model_facts(model),
        "provenance": provenance,
        "provenance_gaps": gaps,
        "provenance_score": {
            "recorded": expected_count(quantized=quantized) - len(gaps),
            "expected": expected_count(quantized=quantized),
        },
        "build_inputs": inputs,
        "notes": notes,
        "fingerprint": fp.to_dict(),
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        msg = f"Cannot read manifest {path}: {e}"
        raise ManifestError(msg) from e
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        msg = f"{path} is not a {SCHEMA} manifest"
        raise ManifestError(msg)
    fp = manifest.get("fingerprint")
    if not isinstance(fp, dict) or fp.get("spec") != canon.SPEC_ID:
        msg = f"{path} carries a fingerprint spec ggufkit does not understand"
        raise ManifestError(msg)
    return manifest


def save_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def default_manifest_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    match = SHARD_RE.match(p.name)
    stem = match["stem"] if match else p.name.removesuffix(".gguf")
    return p.with_name(f"{stem}.manifest.json")


def human_count(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)


def human_bytes(n: int) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def format_summary(manifest: Mapping[str, Any]) -> list[str]:
    m = manifest["model"]
    fp = manifest["fingerprint"]
    prov = manifest["provenance"]
    recorded = sorted(
        k
        for k in prov
        if k not in ("name", "basename", "size_label", "tags", "languages", "description")
    )
    score = manifest["provenance_score"]
    lines = [
        (
            f"{'model':<12} {m.get('architecture') or '?'} {m.get('file_type') or '?'}, "
            f"{manifest['gguf']['n_tensors']} tensors, "
            f"{human_count(m['parameter_count'])} parameters, {human_bytes(m['tensor_bytes'])}"
        ),
        f"{'fingerprint':<12} {fp.get('overall') or '(weights not hashed)'}",
        f"{'provenance':<12} {', '.join(recorded) if recorded else '(nothing recorded)'}",
        f"{'score':<12} {score['recorded']}/{score['expected']} of what a verifiable file records",
    ]
    if manifest["provenance_gaps"]:
        lines.append(f"{'gaps':<12} " + ", ".join(manifest["provenance_gaps"]))
    return lines


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "manifest",
        help="write a provenance manifest: fingerprint, file hashes, origin, gaps",
        description=(
            "Write a ggufkit-manifest/1 JSON file for a model: its fingerprint, file sizes "
            "(and hashes with --file-sha256), what the metadata says about origin and licence, "
            "the build inputs you supply, and the list of provenance a verifiable file would "
            "record but this one does not."
        ),
    )
    p.add_argument("model", type=Path, help="GGUF file, or any shard of a split model")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="manifest path (default: <model>.manifest.json next to the model; '-' for stdout only)",
    )
    p.add_argument("--json", action="store_true", help="also print the manifest to stdout")
    p.add_argument("--file-sha256", action="store_true", help="also hash each file's raw bytes")
    p.add_argument(
        "--input",
        action="append",
        metavar="KEY=VALUE",
        help=f"record a build input ({', '.join(BUILD_INPUT_KEYS)} or any key); repeatable",
    )
    p.add_argument("--note", help="free-text note stored in the manifest")
    p.add_argument("--no-data", action="store_true", help="skip tensor bytes (no weights digest)")
    p.add_argument("--no-per-tensor", action="store_true", help="omit per-tensor digests")
    p.add_argument("-j", "--jobs", type=int, default=default_jobs(), help="hashing threads")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    inputs = parse_build_inputs(args.input)
    to_stdout_only = args.output is not None and str(args.output) == "-"
    target = None if to_stdout_only else (args.output or default_manifest_path(args.model))
    if target is not None and target.exists() and not args.force:
        msg = f"{target} already exists (use --force to overwrite)"
        raise FileExistsError(msg)
    with open_model(args.model) as model:
        manifest = build_manifest(
            model,
            file_sha256=args.file_sha256,
            build_inputs=inputs,
            notes=args.note,
            data=not args.no_data,
            per_tensor=not args.no_per_tensor,
            jobs=max(1, args.jobs),
        )
    if target is not None:
        save_manifest(target, manifest)
    if args.json or to_stdout_only:
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif not args.quiet:
        if target is not None:
            print(f"{'manifest':<12} {target}")
        print("\n".join(format_summary(manifest)))
    return 0
