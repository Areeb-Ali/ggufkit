# Security policy

## Supported versions

Security fixes go into the latest release only.

## What ggufkit does with a model file

ggufkit reads GGUF files you give it, often files downloaded from strangers. It:

- parses them with llama.cpp's own Python package, [`gguf`](https://pypi.org/project/gguf/),
  through a read-only memory map;
- never executes anything stored in a model file: no pickles, no templates, no code;
- loads PyTorch checkpoints (`viz tensor --model-type torch`) only with
  `torch.load(..., weights_only=True)`;
- writes only the files you name with `-o`, `--intoto` or the manifest's default path, and
  refuses to overwrite without `--force`.

A malformed file should produce an error, not a crash, a hang or unbounded memory use. If you
find a file that does otherwise, that is a bug worth reporting, and possibly a security issue.

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/MysterionRise/ggufkit/security/advisories/new)
rather than in a public issue. Include the ggufkit and `gguf` versions (`ggufkit --version`,
`pip show gguf`), the command you ran, and, if you can, a minimal file that reproduces it.

If the root cause is in the `gguf` package itself, it also belongs upstream in
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/security); we will help route it.

## Scope of the guarantees

A ggufkit fingerprint proves integrity and equivalence, and a manifest records claims. Neither
proves that a model is safe or that a quantization was built faithfully; see "What this
proves, and what it does not" in the README. Reports that rest on that distinction are welcome
as documentation issues.
