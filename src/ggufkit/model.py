"""Open a GGUF model as one logical object, whether it is a single file or a set of shards.

llama.cpp's ``gguf-split`` writes ``name-00001-of-00003.gguf`` style shards. The first shard
carries the model metadata, every shard carries ``split.*`` bookkeeping keys, and tensors are
spread across shards in order. :func:`open_model` hides that: it finds the sibling shards,
merges their metadata (dropping the ``split.*`` keys) and concatenates their tensors, so the
rest of ggufkit can treat split and unsplit models identically.
"""

from __future__ import annotations

import gc
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from gguf import GGUFEndian, GGUFReader
from gguf.constants import Keys

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gguf.gguf_reader import ReaderField, ReaderTensor

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)

#: Bookkeeping entries the reader synthesizes from the header. They are not metadata.
PSEUDO_FIELDS = frozenset({"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"})

#: Keys that describe how a model was split into files, not the model itself.
SPLIT_FIELDS = frozenset(
    {
        Keys.Split.LLM_KV_SPLIT_NO,
        Keys.Split.LLM_KV_SPLIT_COUNT,
        Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT,
    }
)


class ModelError(ValueError):
    """Raised when a model cannot be opened as a consistent whole."""


def shard_paths(path: str | Path) -> list[Path]:
    """Return every file of the model ``path`` belongs to, in shard order.

    A file whose name does not follow the ``-NNNNN-of-MMMMM.gguf`` convention is returned
    on its own. Missing sibling shards are an error rather than a silent partial model.
    """
    path = Path(path)
    match = SHARD_RE.match(path.name)
    if match is None:
        return [path]
    count = int(match["count"])
    stem = match["stem"]
    paths = [path.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        msg = f"{path.name} is shard {int(match['no'])} of {count} but these shards are missing: {', '.join(missing)}"
        raise ModelError(msg)
    return paths


@dataclass
class Model:
    """A GGUF model, possibly spread over several shard files, seen as one object."""

    paths: list[Path]
    readers: list[GGUFReader] = field(repr=False)
    fields: OrderedDict[str, ReaderField] = field(repr=False)
    tensors: list[ReaderTensor] = field(repr=False)

    @property
    def path(self) -> Path:
        """The first (or only) file of the model."""
        return self.paths[0]

    @property
    def n_shards(self) -> int:
        return len(self.paths)

    @property
    def version(self) -> int:
        return int(self.readers[0].fields["GGUF.version"].parts[-1][0])

    @property
    def endianess(self) -> GGUFEndian:
        return self.readers[0].endianess

    @property
    def alignment(self) -> int:
        return int(self.readers[0].alignment)

    @property
    def architecture(self) -> str | None:
        f = self.fields.get(Keys.General.ARCHITECTURE)
        return None if f is None else str(f.contents())

    def metadata_fields(self) -> list[ReaderField]:
        """Model metadata only: no reader pseudo-fields, no ``split.*`` bookkeeping."""
        return [
            f
            for name, f in self.fields.items()
            if name not in PSEUDO_FIELDS and name not in SPLIT_FIELDS and f.types
        ]

    def tensor(self, name: str) -> ReaderTensor:
        for t in self.tensors:
            if t.name == name:
                return t
        msg = f"No tensor named {name!r}"
        raise KeyError(msg)

    def tensor_names(self) -> Iterator[str]:
        return (t.name for t in self.tensors)

    def close(self) -> None:
        """Drop references to the memory maps so the files can be released."""
        self.tensors = []
        self.fields = OrderedDict()
        self.readers = []
        gc.collect()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_model(path: str | Path) -> Model:
    """Open ``path`` and any sibling shards as a single :class:`Model`."""
    paths = shard_paths(path)
    readers = [GGUFReader(p, "r") for p in paths]

    endians = {r.endianess for r in readers}
    if len(endians) > 1:
        msg = "Shards disagree on byte order"
        raise ModelError(msg)

    declared = readers[0].fields.get(Keys.Split.LLM_KV_SPLIT_COUNT)
    if declared is not None:
        count = int(declared.contents())
        if count != len(paths):
            msg = (
                f"{paths[0].name} declares {count} shards but {len(paths)} were found; "
                "shards must sit next to each other and follow the -NNNNN-of-MMMMM.gguf naming"
            )
            raise ModelError(msg)

    fields: OrderedDict[str, ReaderField] = OrderedDict()
    for reader in readers:
        for name, f in reader.fields.items():
            fields.setdefault(name, f)

    tensors = [t for reader in readers for t in reader.tensors]
    seen: set[str] = set()
    for t in tensors:
        if t.name in seen:
            msg = f"Tensor {t.name!r} appears in more than one shard"
            raise ModelError(msg)
        seen.add(t.name)

    return Model(paths=paths, readers=readers, fields=fields, tensors=tensors)
